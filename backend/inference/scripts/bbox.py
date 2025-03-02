import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import argparse  # Add this import
import sys
import torch
import threading
from queue import Queue

def get_optimal_window_size(frame_width, frame_height):
    # Get screen dimensions using tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Calculate maximum window size (leaving some margin for window decorations)
    max_width = int(screen_width * 0.9)
    max_height = int(screen_height * 0.9)
    
    # Calculate scaling factor
    scale_width = max_width / frame_width
    scale_height = max_height / frame_height
    scale = min(scale_width, scale_height, 1.0)  # Don't upscale if video is smaller than screen
    
    return int(frame_width * scale), int(frame_height * scale), scale

def select_files(): 
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Ask for video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
    )
    
    if not video_path:
        messagebox.showerror("Error", "No video file selected!")
        return None, None

    # Set default save location
    default_save_dir = r"C:\Users\higaz\Samurai\input"
    os.makedirs(default_save_dir, exist_ok=True)
    
    # Generate default filename from video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    default_save_path = os.path.join(default_save_dir, f"{video_name}_bbox.txt")
    
    # Ask for save location
    txt_path = filedialog.asksaveasfilename(
        initialdir=default_save_dir,
        initialfile=f"{video_name}_bbox.txt",
        title="Save Bounding Box Coordinates",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        defaultextension=".txt"
    )
    
    if not txt_path:
        messagebox.showerror("Error", "No save location selected!")
        return None, None

    return video_path, txt_path

def preview_existing_bbox(video_path, bbox_path):
    """Preview existing bbox on first frame"""
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        messagebox.showerror("Error", "Could not read video file!")
        return

    # Read bbox coordinates
    try:
        with open(bbox_path, 'r') as f:
            line = f.read().strip()
            if '[' in line:  # Handle [x,y,w,h],1 format
                coords = line.split('],')[0].strip('[').split(',')
            else:  # Handle space-separated format
                coords = line.split()
            x, y, w, h = map(float, coords[:4])
    except:
        messagebox.showerror("Error", f"Could not read bbox file: {bbox_path}")
        return

    # Calculate window size
    height, width = frame.shape[:2]
    window_width, window_height, scale = get_optimal_window_size(width, height)
    
    # Scale frame and coordinates
    display_frame = cv2.resize(frame, (window_width, window_height))
    display_x = int(x * scale)
    display_y = int(y * scale)
    display_w = int(w * scale)
    display_h = int(h * scale)

    # Draw bbox
    cv2.rectangle(display_frame, 
                 (display_x, display_y), 
                 (display_x + display_w, display_y + display_h), 
                 (0, 255, 0), 2)

    # Show coordinates
    text = f'x:{int(x)}, y:{int(y)}, w:{int(w)}, h:{int(h)}'
    cv2.putText(display_frame, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    window_name = 'Existing Bounding Box (Press any key to continue)'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)
    cv2.imshow(window_name, display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_bbox_coordinates(video_path, output_txt_path):
    global current_bbox, current_video_path
    current_video_path = video_path
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to read video!")
        return
    
    # Get original dimensions
    original_height, original_width = frame.shape[:2]
    
    # Calculate optimal window size and scale
    window_width, window_height, scale = get_optimal_window_size(original_width, original_height)
    
    # Resize frame for display
    display_frame = cv2.resize(frame, (window_width, window_height))
    temp_frame = display_frame.copy()
    
    # Create window and set mouse callback
    bbox = []
    drawing = False
    start_point = None
    current_coords = "No coordinates yet"  # Store current coordinates for display
    
    def show_menu():
        menu = tk.Tk()
        menu.title("Bounding Box Options")
        
        def load_existing():
            menu.destroy()
            if os.path.exists(output_txt_path):
                with open(output_txt_path, 'r') as f:
                    line = f.read().strip()
                    if '[' in line:
                        coords = line.split('],')[0].strip('[').split(',')
                    else:
                        coords = line.split()
                    x, y, w, h = map(float, coords[:4])
                    nonlocal bbox, temp_frame, current_coords
                    bbox = [int(x), int(y), int(w), int(h)]
                    # Draw loaded bbox
                    display_start = (int(x * scale), int(y * scale))
                    display_end = (int((x + w) * scale), int((y + h) * scale))
                    temp_frame = display_frame.copy()
                    cv2.rectangle(temp_frame, display_start, display_end, (0, 255, 0), 2)
                    current_coords = f"Loaded: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}"
            else:
                messagebox.showerror("Error", "No existing bbox file found!")

        def start_new():
            menu.destroy()
            nonlocal temp_frame, bbox, current_coords
            temp_frame = display_frame.copy()
            bbox = []
            current_coords = "No coordinates yet"

        tk.Button(menu, text="Load Existing Bbox", command=load_existing).pack(pady=5)
        tk.Button(menu, text="Create New Bbox", command=start_new).pack(pady=5)
        menu.mainloop()

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, temp_frame, bbox, current_coords
        
        def display_to_original(x, y):
            return int(x / scale), int(y / scale)
        
        def original_to_display(x, y):
            return int(x * scale), int(y * scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            temp_frame = display_frame.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_frame = temp_frame.copy()
            # Draw guide lines
            cv2.line(current_frame, (x, 0), (x, window_height), (200,200,200), 1)
            cv2.line(current_frame, (0, y), (window_width, y), (200,200,200), 1)
            
            # Calculate and display current coordinates
            start_x, start_y = display_to_original(min(start_point[0], x), 
                                                 min(start_point[1], y))
            end_x, end_y = display_to_original(max(start_point[0], x), 
                                             max(start_point[1], y))
            w = end_x - start_x
            h = end_y - start_y
            current_coords = f"x={start_x}, y={start_y}, w={w}, h={h}"
            
            # Draw rectangle and coordinates
            cv2.rectangle(current_frame, start_point, (x, y), (0,255,0), 2)
            cv2.putText(current_frame, current_coords, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Draw Bounding Box (S to save, Q to quit, M for menu)', current_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            
            # Convert coordinates
            start_x, start_y = display_to_original(min(start_point[0], end_point[0]), 
                                                 min(start_point[1], end_point[1]))
            end_x, end_y = display_to_original(max(start_point[0], end_point[0]), 
                                             max(start_point[1], end_point[1]))
            
            w = end_x - start_x
            h = end_y - start_y
            
            bbox = [start_x, start_y, w, h]
            current_coords = f"x={start_x}, y={start_y}, w={w}, h={h}"
            
            # Update display frame with final bbox
            temp_frame = display_frame.copy()
            display_start = original_to_display(start_x, start_y)
            display_end = original_to_display(start_x + w, start_y + h)
            cv2.rectangle(temp_frame, display_start, display_end, (0,255,0), 2)
            cv2.putText(temp_frame, current_coords, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Draw Bounding Box (S to save, Q to quit, M for menu)', temp_frame)
            # Request mask update in background
            # When bbox is created/loaded, store it globally
            current_bbox = bbox
    
    window_name = 'Draw Bounding Box (S to save, Q to quit, M for menu)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Initial window positioning
    cv2.moveWindow(window_name, 0, 0)
    
    # Show menu at start
    show_menu()
    
    while True:
        # Display frame with current coordinates
        frame_to_show = temp_frame.copy()
        cv2.putText(frame_to_show, current_coords, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow(window_name, frame_to_show)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and bbox:
            with open(output_txt_path, 'w') as f:
                f.write(f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}],1")
            messagebox.showinfo("Success", 
                              f"Saved bbox coordinates to:\n{output_txt_path}\n\n"
                              f"Original resolution: {original_width}x{original_height}\n"
                              f"Coordinates saved: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
            break
        elif key == ord('q'):
            break
        elif key == ord('m'):
            show_menu()
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    
    class BBoxTool:
        def __init__(self, root):
            print("Initializing BBoxTool...")  # Debug
            self.root = root
            self.root.title("Bounding Box Tool")

            self.toolbar = tk.Frame(root)
            self.toolbar.pack(side="top", fill="x")

            self.predictor = None
            self.state = None

            # Add paths
            self.sam2_path = r"C:\Users\higaz\Samurai\samurai\sam2"
            self.scripts_dir = r"C:\Users\higaz\Samurai\samurai\scripts"
            sys.path.append(self.scripts_dir)
            sys.path.append(self.sam2_path)
            print("BBoxTool initialized with predictor and state as None")  # Debug
            self.display_frame = None  # Initialize display_frame

            self.current_mask = None

            self.update_queue = Queue()

            # Add preview checkbox to toolbar
            self.show_preview = tk.BooleanVar(value=False)
            self.preview_checkbox = tk.Checkbutton(
                self.toolbar, 
                text="Live Mask Preview", 
                variable=self.show_preview,
                command=self.on_preview_toggle  # Add this callback
            )
            self.preview_checkbox.pack(side="left", padx=2, pady=2)
            # Start background thread for mask updates
            self.mask_thread = threading.Thread(target=self.mask_update_worker, daemon=True)
            self.mask_thread.start()
            
            # Add toolbar frame
            self.toolbar = tk.Frame(root)
            self.toolbar.pack(side="top", fill="x")
            
            # Initialize mode
            self.current_mode = "bbox"  # Default to bbox mode
            
            # Create toolbar buttons
            self.create_toolbar()
            
            # Initialize other variables
            self.current_video_path = None
            self.current_bbox_path = None
            self.current_bbox = None
            self.video_frame = None

            self.points = {'positive': [], 'negative': []}
            self.points_history = []  # Add this line
            
            self.drawing = False
            self.window_name = 'Bounding Box View'
            print("Finished initialization")  # Debug
            
            # Create menu bar
            self.menubar = tk.Menu(root)
            self.root.config(menu=self.menubar)
            
            # Create File menu
            self.file_menu = tk.Menu(self.menubar, tearoff=0)
            self.menubar.add_cascade(label="File", menu=self.file_menu)
            self.file_menu.add_command(label="New Bounding Box", command=self.create_new_bbox)
            self.file_menu.add_command(label="Load Existing", command=self.load_existing_bbox)
            self.file_menu.add_command(label="Save As...", command=self.save_bbox_as)
            self.file_menu.add_separator()
            self.file_menu.add_command(label="Generate Masks", command=self.generate_masks)
            self.file_menu.add_separator()
            self.file_menu.add_command(label="Exit", command=self.root.quit)

            # Create Operations menu
            self.ops_menu = tk.Menu(self.menubar, tearoff=0)
            self.menubar.add_cascade(label="Operations", menu=self.ops_menu)
            self.ops_menu.add_command(label="Generate Masks", command=self.generate_masks)

            # Add keyboard shortcuts
            self.root.bind('<Control-s>', lambda e: self.save_bbox_as())
            self.root.bind('<Control-g>', lambda e: self.generate_masks())  # New shortcut for generate masks
            
            # Welcome message
            self.welcome_label = tk.Label(root, 
                text="Welcome to the Bounding Box Tool\nUse the File menu to get started",
                justify=tk.CENTER, pady=20)
            self.welcome_label.pack()

            # Add keyboard shortcuts
            self.root.bind('<Control-s>', lambda e: self.save_bbox_as())
            self.root.bind('<s>', lambda e: self.quick_save())
            # Add undo shortcut
            self.root.bind('<Control-z>', lambda e: self.undo_last_action())

        def set_model(self, predictor, state):
            """Set the SAM predictor and state"""
            print(f"set_model called with predictor: {predictor is not None}, state: {state is not None}")  # Debug
            self.predictor = predictor
            self.state = state
            print(f"After set_model - predictor: {self.predictor is not None}, state: {self.state is not None}")  # Debug


        def on_preview_toggle(self):
            """Debug callback for preview checkbox"""
            print(f"Preview toggled: {self.show_preview.get()}")  # Debug print
            if self.show_preview.get():
                self.request_mask_update()  # Request initial update when enabled

        def request_mask_update(self):
            """Request a mask update in the background"""
            if self.show_preview.get():
                self.update_queue.put(True)
                print("Mask update requested")  # Debug

        def mask_update_worker(self):
            """Background worker that processes mask updates"""
            while True:
                try:
                    # Wait for update request
                    print("Waiting for update request...")  # Debug
                    update_request = self.update_queue.get()
                    if update_request:
                        print("Got update request")  # Debug
                        self.request_live_mask()  # Call to generate the mask
                except Exception as e:
                    print(f"Mask update error: {str(e)}")
                    import traceback
                    traceback.print_exc()

        def create_toolbar(self):
            # Create simple colored buttons (you can replace with proper icons)
            bbox_btn = tk.Button(self.toolbar, 
                               text="📦 Box", 
                               bg="lightgray",
                               command=lambda: self.set_mode("bbox"))
            bbox_btn.pack(side="left", padx=2, pady=2)
            
            pos_btn = tk.Button(self.toolbar, 
                              text="➕ Positive", 
                              bg="lightgreen",
                              command=lambda: self.set_mode("positive"))
            pos_btn.pack(side="left", padx=2, pady=2)
            
            neg_btn = tk.Button(self.toolbar, 
                              text="❌ Negative", 
                              bg="pink",
                              command=lambda: self.set_mode("negative"))
            neg_btn.pack(side="left", padx=2, pady=2)

            undo_btn = tk.Button(self.toolbar,
                               text="↩ Undo",
                               bg="lightblue", 
                               command=self.undo_last_action)
            undo_btn.pack(side="left", padx=2, pady=2)
            
            # Add mode label
            self.mode_label = tk.Label(self.toolbar, text="Mode: Bounding Box")
            self.mode_label.pack(side="left", padx=10)

        def undo_last_action(self):
            """Undo the last action"""
            if not self.points_history:
                messagebox.showinfo("Info", "Nothing to undo!")
                return
                
            last_action = self.points_history.pop()
            if last_action['type'] == 'point':
                point_type = last_action['point_type']
                if self.points[point_type]:
                    self.points[point_type].pop()
            
            # Update display
            self.temp_frame = self.display_frame.copy()
            self.draw_current_points(self.temp_frame)
            cv2.imshow(self.window_name, self.temp_frame)
            self.request_mask_update()  # <-- Add this

        def set_mode(self, mode):
            print(f"Setting mode to: {mode}")  # Debug
            self.current_mode = mode
            mode_names = {
                "bbox": "Bounding Box",
                "positive": "Positive Points",
                "negative": "Negative Points"
            }
            self.mode_label.config(text=f"Mode: {mode_names[mode]}")
            print(f"Current mode is now: {self.current_mode}")  # Debug
            
            # Update window title if window exists
            try:
                if hasattr(self, 'window_name') and self.window_name:
                    cv2.setWindowTitle(self.window_name, f'Bounding Box View - {mode_names[mode]} Mode')
            except Exception as e:
                print(f"Could not update window title: {e}")

        def mouse_callback(self, event, x, y, flags, param):
            try:
                print(f"Current mode: {self.current_mode}, Event: {event}")  # Debug print

                # Get the actual mode value from StringVar
                current_mode = self.current_mode

                if self.current_mode == "positive" and event == cv2.EVENT_LBUTTONDOWN:
                    # Add positive point
                    orig_x, orig_y = self.display_to_original(x, y)
                    self.points['positive'].append((orig_x, orig_y))
                    print(f"Added positive point at ({orig_x}, {orig_y})")  # Debug
                    
                    # Update display
                    self.temp_frame = self.display_frame.copy()
                    self.draw_current_points(self.temp_frame)
                    self.update_display()
                    self.request_mask_update()  # <-- Add this


                elif self.current_mode == "negative" and event == cv2.EVENT_LBUTTONDOWN:
                    # Add negative point
                    orig_x, orig_y = self.display_to_original(x, y)
                    self.points['negative'].append((orig_x, orig_y))
                    print(f"Added negative point at ({orig_x}, {orig_y})")  # Debug
                    
                    # Update display
                    self.temp_frame = self.display_frame.copy()
                    self.draw_current_points(self.temp_frame)
                    self.update_display()
                    self.request_mask_update()  # <-- Add this

                elif self.current_mode == "bbox":
                    # Your existing bbox handling code
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.drawing = True
                        self.start_point = (x, y)
                        self.temp_frame = self.display_frame.copy()
                        self.draw_current_points(self.temp_frame)
                        self.request_mask_update()  # <-- Add this

                    elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                        frame_copy = self.display_frame.copy()
                        self.draw_current_points(frame_copy)
                        cv2.rectangle(frame_copy, self.start_point, (x, y), (0, 255, 0), 2)
                        cv2.line(frame_copy, (x, 0), (x, frame_copy.shape[0]), (200,200,200), 1)
                        cv2.line(frame_copy, (0, y), (frame_copy.shape[1], y), (200,200,200), 1)
                        
                        # Calculate coordinates
                        start_x, start_y = self.display_to_original(min(self.start_point[0], x), 
                                                                  min(self.start_point[1], y))
                        end_x, end_y = self.display_to_original(max(self.start_point[0], x), 
                                                              max(self.start_point[1], y))
                        w = end_x - start_x
                        h = end_y - start_y
                        coords_text = f"x={start_x}, y={start_y}, w={w}, h={h}"
                        points_text = f"Pos points: {len(self.points['positive'])} Neg points: {len(self.points['negative'])}"
                        cv2.putText(frame_copy, coords_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame_copy, points_text, (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, frame_copy)
                        self.request_mask_update()  # <-- Add this

                    elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                        self.drawing = False
                        end_point = (x, y)
                        
                        # Convert coordinates
                        start_x, start_y = self.display_to_original(min(self.start_point[0], end_point[0]), 
                                                                  min(self.start_point[1], end_point[1]))
                        end_x, end_y = self.display_to_original(max(self.start_point[0], end_point[0]), 
                                                              max(self.start_point[1], end_point[1]))
                        
                        w = end_x - start_x
                        h = end_y - start_y
                        
                        if w < 10 or h < 10:
                            messagebox.showwarning("Warning", "Bounding box too small!")
                            return
                        
                        self.current_bbox = [start_x, start_y, w, h]
                        self.draw_bbox()
                        self.request_mask_update()  # <-- Add this

            except Exception as e:
                print(f"Error in mouse_callback: {str(e)}")
                import traceback
                print(traceback.format_exc())

        def draw_current_points(self, frame):
            # Draw existing points
            for point in self.points['positive']:
                disp_x = int(point[0] * self.scale)
                disp_y = int(point[1] * self.scale)
                cv2.circle(frame, (disp_x, disp_y), 5, (0, 255, 0), -1)  # Green for positive
            
            for point in self.points['negative']:
                disp_x = int(point[0] * self.scale)
                disp_y = int(point[1] * self.scale)
                cv2.circle(frame, (disp_x, disp_y), 5, (0, 0, 255), -1)  # Red for negative

            # Draw current bbox if it exists
            if self.current_bbox:
                x, y, w, h = self.current_bbox
                display_start = (int(x * self.scale), int(y * self.scale))
                display_end = (int((x + w) * self.scale), int((y + h) * self.scale))
                cv2.rectangle(frame, display_start, display_end, (0, 255, 0), 2)

        def draw_coordinate_info(self, frame, x, y):
            if self.start_point:
                start_x, start_y = self.display_to_original(min(self.start_point[0], x), 
                                                          min(self.start_point[1], y))
                end_x, end_y = self.display_to_original(max(self.start_point[0], x), 
                                                      max(self.start_point[1], y))
                w = end_x - start_x
                h = end_y - start_y
                coords_text = f"x={start_x}, y={start_y}, w={w}, h={h}"
                points_text = f"Pos points: {len(self.points['positive'])} Neg points: {len(self.points['negative'])}"
                cv2.putText(frame, coords_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, points_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        def create_new_bbox(self):
            print("\n=== Starting create_new_bbox ===")  # Debug
            print("Opening file dialog for video selection...")  # Debug
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
            )
            print(f"Selected video path: {video_path}")  # Debug
            
            if not video_path:
                print("No video path selected, returning")  # Debug
                return

            # print("Opening file dialog for bbox save location...")  # Debug
            # try:
            #     txt_path = filedialog.asksaveasfilename(
            #         title="Save Bounding Box As",
            #         filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            #         defaultextension=".txt"
            #     )
            #     print(f"Selected save path: {txt_path}")  # Debug
            # except Exception as e:
            #     print(f"Error in save dialog: {str(e)}")
            #     return

            # if not txt_path:
            #     print("No save path selected, returning")  # Debug
            #     return

            try:
                print("\n=== Setting up video and bbox ===")  # Debug
                print(f"Setting current_video_path to: {video_path}")  # Debug
                self.current_video_path = video_path
                
                print("\n=== Loading video frame ===")  # Debug
                self.load_video_frame()
                
                print("\n=== Setting up bbox drawing ===")  # Debug
                self.setup_bbox_drawing()
                
                print("Updating menus...")  # Debug
                self.update_menus()
                
            except Exception as e:
                print(f"\nERROR in create_new_bbox: {str(e)}")  # Debug
                import traceback
                print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to create new bbox: {str(e)}")

        def load_existing_bbox(self):
            print("Starting load_existing_bbox...")  # Debug print
            
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
            )
            
            print(f"Selected video path: {video_path}")  # Debug print
            
            if not video_path:
                print("No video path selected")  # Debug print
                return

            bbox_path = filedialog.askopenfilename(
                title="Select Existing Bbox File",
                filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
            )

            print(f"Selected bbox path: {bbox_path}")  # Debug print

            if not bbox_path:
                print("No bbox path selected")  # Debug print
                return

            try:
                print("Setting paths and loading video frame...")  # Debug
                self.current_video_path = video_path
                self.current_bbox_path = bbox_path
                print("Loading video frame...")  # Debug print
                self.load_video_frame()
                print("Loading bbox data...")  # Debug print
                self.load_existing_bbox_data()
                print("Updating menus...")  # Debug print
                self.update_menus()
            except Exception as e:
                print(f"Error in load_existing_bbox: {str(e)}")  # Debug print
                import traceback
                print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to load: {str(e)}")

        def load_video_frame(self):
            print("\n=== Starting load_video_frame ===")  # Debug
            try:
                print(f"Opening video file: {self.current_video_path}")  # Debug
                cap = cv2.VideoCapture(self.current_video_path)
                print(f"VideoCapture object created: {cap}")  # Debug
                print(f"VideoCapture isOpened: {cap.isOpened()}")  # Debug
                
                print("Reading first frame...")  # Debug
                ret, frame = cap.read()
                print(f"Read success: {ret}")  # Debug
                if frame is not None:
                    print(f"Frame shape: {frame.shape}")  # Debug
                else:
                    print("Frame is None!")  # Debug
                    
                print("Releasing video capture...")  # Debug
                cap.release()
                
                if not ret:
                    print("Failed to read video frame")  # Debug
                    raise Exception("Could not read video file")
                
                print("Setting up frame display...")  # Debug
                self.original_frame = frame
                height, width = frame.shape[:2]
                print(f"Original dimensions: {width}x{height}")  # Debug
                
                window_width, window_height, self.scale = get_optimal_window_size(width, height)
                print(f"Calculated window size: {window_width}x{window_height}, scale: {self.scale}")  # Debug
                
                print("Resizing frame for display...")  # Debug
                self.display_frame = cv2.resize(frame, (window_width, window_height))
                self.temp_frame = self.display_frame.copy()
                print("Successfully loaded and prepared video frame")  # Debug
                
            except Exception as e:
                print(f"\nERROR in load_video_frame: {str(e)}")  # Debug
                import traceback
                print(traceback.format_exc())
                raise

        def load_existing_bbox_data(self):
            print(f"Attempting to load bbox from: {self.current_bbox_path}")  # Debug print
            try:
                with open(self.current_bbox_path, 'r') as f:
                    lines = f.readlines()
                    # First line is bbox
                    line = lines[0].strip()
                    if '[' in line:
                        coords = line.split('],')[0].strip('[').split(',')
                    else:
                        coords = line.split()
                    print(f"Parsed coordinates: {coords}")  # Debug print
                    x, y, w, h = map(float, coords[:4])
                    self.current_bbox = [int(x), int(y), int(w), int(h)]
                    print(f"Set current_bbox to: {self.current_bbox}")  # Debug print
                    
                    # Reset points
                    self.points = {'positive': [], 'negative': []}
                    
                    # Look for points
                    if len(lines) > 1 and lines[1].strip() == "POINTS":
                        for line in lines[2:]:
                            type_, px, py = line.strip().split(',')
                            point = (int(float(px)), int(float(py)))
                            if type_ == 'p':
                                self.points['positive'].append(point)
                            elif type_ == 'n':
                                self.points['negative'].append(point)
                    
                    self.draw_bbox()
            except Exception as e:
                print(f"Error in load_existing_bbox_data: {str(e)}")  # Debug print
                import traceback
                print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to load bbox file: {str(e)}")

        def draw_bbox(self):
            print("Starting draw_bbox...")  # Debug print
            if self.current_bbox and self.display_frame is not None:
                print(f"Drawing bbox: {self.current_bbox}")  # Debug print
                self.temp_frame = self.display_frame.copy()
                x, y, w, h = self.current_bbox
                display_start = (int(x * self.scale), int(y * self.scale))
                display_end = (int((x + w) * self.scale), int((y + h) * self.scale))
                print(f"Display coordinates: {display_start} to {display_end}")  # Debug print
                cv2.rectangle(self.temp_frame, display_start, display_end, (0, 255, 0), 2)
                
                # Draw all points
                for point in self.points['positive']:
                    disp_x = int(point[0] * self.scale)
                    disp_y = int(point[1] * self.scale)
                    cv2.circle(self.temp_frame, (disp_x, disp_y), 5, (0, 255, 0), -1)
                
                for point in self.points['negative']:
                    disp_x = int(point[0] * self.scale)
                    disp_y = int(point[1] * self.scale)
                    cv2.circle(self.temp_frame, (disp_x, disp_y), 5, (0, 0, 255), -1)
                
                coords_text = f"x={x}, y={y}, w={w}, h={h}"
                points_text = f"Pos points: {len(self.points['positive'])} Neg points: {len(self.points['negative'])}"
                cv2.putText(self.temp_frame, coords_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(self.temp_frame, points_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.update_display()
            else:
                print(f"Cannot draw bbox: current_bbox={self.current_bbox}, "
                      f"display_frame={'exists' if self.display_frame is not None else 'None'}")  # Debug print

        def setup_bbox_drawing(self):
            self.drawing = False
            self.start_point = None
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.update_display()
        def mouse_callback(self, event, x, y, flags, param):
            try:
                if self.display_frame is None:
                    return

                # Handle positive point mode
                if self.current_mode == "positive" and event == cv2.EVENT_LBUTTONDOWN:
                    orig_x, orig_y = self.display_to_original(x, y)
                    self.points['positive'].append((orig_x, orig_y))
                    self.points_history.append({
                        'type': 'point',
                        'point_type': 'positive'
                    })
                    self.temp_frame = self.display_frame.copy()
                    self.draw_current_points(self.temp_frame)
                    cv2.imshow(self.window_name, self.temp_frame)
                    print(f"Added positive point at ({orig_x}, {orig_y})")
                    self.request_mask_update()

                # Handle negative point mode  
                elif self.current_mode == "negative" and event == cv2.EVENT_LBUTTONDOWN:
                    orig_x, orig_y = self.display_to_original(x, y)
                    self.points['negative'].append((orig_x, orig_y))
                    self.points_history.append({
                        'type': 'point',
                        'point_type': 'negative'
                    })
                    self.temp_frame = self.display_frame.copy()
                    self.draw_current_points(self.temp_frame)
                    cv2.imshow(self.window_name, self.temp_frame)
                    print(f"Added negative point at ({orig_x}, {orig_y})")
                    self.request_mask_update()

                # Handle bbox mode
                elif self.current_mode == "bbox":
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.drawing = True
                        self.start_point = (x, y)
                        self.temp_frame = self.display_frame.copy()
                        self.draw_current_points(self.temp_frame)
                        self.request_mask_update()

                    elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                        frame_copy = self.display_frame.copy()
                        self.draw_current_points(frame_copy)
                        cv2.rectangle(frame_copy, self.start_point, (x, y), (0, 255, 0), 2)
                        cv2.line(frame_copy, (x, 0), (x, frame_copy.shape[0]), (200,200,200), 1)
                        cv2.line(frame_copy, (0, y), (frame_copy.shape[1], y), (200,200,200), 1)
                        
                        # Calculate coordinates
                        start_x, start_y = self.display_to_original(min(self.start_point[0], x), 
                                                                min(self.start_point[1], y))
                        end_x, end_y = self.display_to_original(max(self.start_point[0], x), 
                                                            max(self.start_point[1], y))
                        w = end_x - start_x
                        h = end_y - start_y
                        
                        # Display coordinate info
                        coords_text = f"x={start_x}, y={start_y}, w={w}, h={h}"
                        points_text = f"Pos points: {len(self.points['positive'])} Neg points: {len(self.points['negative'])}"
                        cv2.putText(frame_copy, coords_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame_copy, points_text, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, frame_copy)

                    elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                        self.drawing = False
                        end_point = (x, y)
                        
                        # Convert coordinates
                        start_x, start_y = self.display_to_original(min(self.start_point[0], end_point[0]), 
                                                                min(self.start_point[1], end_point[1]))
                        end_x, end_y = self.display_to_original(max(self.start_point[0], end_point[0]), 
                                                            max(self.start_point[1], end_point[1]))
                        
                        w = end_x - start_x
                        h = end_y - start_y
                        
                        # Validate bbox size
                        if w < 10 or h < 10:
                            messagebox.showwarning("Warning", "Bounding box too small!")
                            return
                        
                        self.current_bbox = [start_x, start_y, w, h]
                        self.draw_bbox()
                        self.request_mask_update()

            except Exception as e:
                print(f"Mouse callback error: {str(e)}")
                import traceback
                traceback.print_exc()

        def display_to_original(self, x, y):
            return int(x / self.scale), int(y / self.scale)

        def update_display(self):
            if self.temp_frame is not None:
                cv2.imshow(self.window_name, self.temp_frame)
                cv2.waitKey(1)

        def update_display_with_mask(self):
            """Update display with current frame and mask overlay"""
            if self.display_frame is None:
                return
        
            try:
                print("=== Starting update_display_with_mask ===")
                # Create display with current frame
                display = self.display_frame.copy()
                print("Creating display copy...")
                
                # Add mask overlay if available
                if self.current_mask is not None and self.show_preview.get():
                    # Squeeze the mask to remove any singleton dimensions
                    mask_squeezed = np.squeeze(self.current_mask)
                    print(f"Original mask shape: {mask_squeezed.shape}")
                    
                    # Scale the mask to match the display dimensions
                    scaled_mask = cv2.resize(mask_squeezed, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
                    print(f"Scaled mask shape: {scaled_mask.shape}")
                    
                    # Check mask values
                    unique_values = np.unique(scaled_mask)
                    print(f"Unique values in mask: {unique_values}")
                    
                    # Threshold the mask to create a binary mask
                    threshold_value = 0  # Adjust this value based on your mask's range
                    binary_mask = (scaled_mask > threshold_value).astype(np.uint8)
                    print(f"Binary mask unique values: {np.unique(binary_mask)}")
                    
                    # Ensure mask dimensions match the display dimensions
                    if binary_mask.shape == display.shape[:2]:
                        mask_overlay = np.zeros_like(display)
                        mask_overlay[binary_mask == 1] = [0, 0, 255]  # Red overlay
                        alpha = 0.5
                        display = cv2.addWeighted(display, 1, mask_overlay, alpha, 0)
                    else:
                        print(f"Binary mask shape {binary_mask.shape} does not match display shape {display.shape[:2]}")
            
                # Draw points and bbox
                self.draw_current_points(display)
                
                # Update display
                cv2.imshow(self.window_name, display)
    
            except Exception as e:
                print(f"Display update error: {str(e)}")
                import traceback
                traceback.print_exc()

        def save_bbox_as(self):
            if self.current_bbox is None:
                return
            
            # Get original image dimensions
            height, width = self.original_frame.shape[:2]
            
            # Ensure coordinates are in the correct format
            x, y, w, h = self.current_bbox
            
            # Add validation
            if w > width or h > height:
                messagebox.showwarning("Warning", "Bounding box dimensions exceed image size!")
                return
            
            save_path = filedialog.asksaveasfilename(
                title="Save Bounding Box and Points As",
                filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
                defaultextension=".txt"
            )
            
            if save_path:
                with open(save_path, 'w') as f:
                    # Save bbox and label
                    x, y, w, h = self.current_bbox
                    f.write(f"[{x},{y},{w},{h}],1\n")
                    # Save points
                    f.write("POINTS\n")
                    for px, py in self.points['positive']:
                        f.write(f"p,{px},{py}\n")
                    for px, py in self.points['negative']:
                        f.write(f"n,{px},{py}\n")
                messagebox.showinfo("Success", 
                          f"Saved bbox coordinates to:\n{save_path}\n\n"
                          f"Original resolution: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}\n"
                          f"Coordinates saved: x={x}, y={y}, w={w}, h={h}")
        
        def request_live_mask(self):
            """Generate a single mask for live preview"""
            print("\n=== Starting request_live_mask ===")  # Debug
            try:
                print("Importing generate_single_mask...")  # Debug
                from mask_generator import generate_single_mask
                
                print("Setting model path...")  # Debug
                model_path = r"C:\Users\higaz\Samurai\samurai\sam2\checkpoints\sam2.1_hiera_large.pt"
                print(f"Model path: {model_path}")  # Debug
                
                print("Calling generate_single_mask...")  # Debug
                print(f"Current video path: {self.current_video_path}")  # Debug
                print(f"Points: {self.points}")  # Debug
                print(f"Current bbox: {self.current_bbox}")  # Debug
                
                mask = generate_single_mask(
                    self.current_video_path,
                    self.points,
                    self.current_bbox,
                    model_path
                )
                
                if mask is not None:
                    print("Generated mask, updating display")  # Debug
                    # Move mask to CPU and convert to numpy
                    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                    self.current_mask = mask_np
                    self.update_display_with_mask()
                else:
                    print("No mask was generated!")  # Debug
                        
            except Exception as e:
                print("\nERROR in request_live_mask:")  # Debug
                print(f"Live mask generation error: {str(e)}")
                import traceback
                traceback.print_exc()
                print("=== End request_live_mask with error ===\n")  # Debug
            else:
                print("=== Successfully completed request_live_mask ===\n")  # Debug

        def generate_masks(self):
            if not self.current_video_path:
                messagebox.showerror("Error", "Please load a video first!")
                return
            
            # Example bbox and points
            bbox = self.current_bbox
            points = self.points

            try:
                sam2_path = r"C:\Users\higaz\Samurai\samurai\sam2"
                scripts_dir = r"C:\Users\higaz\Samurai\samurai\scripts"
                sys.path.append(scripts_dir)
                sys.path.append(sam2_path)
                
                from demo import main
            except ImportError as e:
                messagebox.showerror("Error", f"Could not import segmentation model: {str(e)}")
                return
            
            mask_path = filedialog.asksaveasfilename(
                title="Save Mask Video As",
                filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
                defaultextension=".mp4",
                initialfile="mask.mp4"
            )
            if not mask_path:
                return

            video_output_path = filedialog.asksaveasfilename(
                title="Save Visualization Video As",
                filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
                defaultextension=".mp4",
                initialfile="visualization.mp4"
            )
            if not video_output_path:
                return

            try:
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Processing")
                progress_window.geometry("300x100")
                progress_window.geometry(f"+{self.root.winfo_x() + 50}+{self.root.winfo_y() + 50}")
                
                label = tk.Label(progress_window, text="Generating masks...\nThis may take a few minutes.")
                label.pack(pady=20)
                self.root.update()

                model_path = r"C:\Users\higaz\Samurai\samurai\sam2\checkpoints\sam2.1_hiera_large.pt"
                args = argparse.Namespace(
                    video_path=self.current_video_path,
                    txt_path=None,  # Not needed anymore
                    model_path=model_path,
                    video_output_path=video_output_path,
                    mask_path=mask_path,
                    save_to_video=True,
                    save_to_mask=True
                )

                # Run the segmentation model with bbox and points
                main(args, bbox=bbox, points=points)

                progress_window.destroy()
                messagebox.showinfo("Success", 
                          f"Processing complete!\n\n"
                          f"Input Video: {self.current_video_path}\n"
                          f"Outputs:\n"
                          f"Mask video: {mask_path}\n"
                          f"Visualization: {video_output_path}")
            
            except Exception as e:
                if 'progress_window' in locals():
                    progress_window.destroy()
                messagebox.showerror("Error", f"Failed to generate masks:\n{str(e)}")
                import traceback
                print("Detailed error:")
                print(traceback.format_exc())

        def quick_save(self):
            """Quick save to current bbox path if it exists"""
            if self.current_bbox_path and self.current_bbox:
                with open(self.current_bbox_path, 'w') as f:
                    f.write(f"[{self.current_bbox[0]},{self.current_bbox[1]},{self.current_bbox[2]},{self.current_bbox[3]}],1")
                messagebox.showinfo("Success", 
                          f"Saved bbox coordinates to:\n{self.current_bbox_path}\n\n"
                          f"Original resolution: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}\n"
                          f"Coordinates saved: x={self.current_bbox[0]}, y={self.current_bbox[1]}, "
                          f"w={self.current_bbox[2]}, h={self.current_bbox[3]}")
            else:
                self.save_bbox_as()

        def update_menus(self):
            """Update menu item states based on current state"""
            if self.current_video_path:
                # Enable generate masks option when video is loaded
                self.file_menu.entryconfig("Generate Masks", state="normal")
                self.ops_menu.entryconfig("Generate Masks", state="normal")
            else:
                # Disable generate masks option when video is missing
                self.file_menu.entryconfig("Generate Masks", state="disabled")
                self.ops_menu.entryconfig("Generate Masks", state="disabled")

    # Create and run application
    root = tk.Tk()
    app = BBoxTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
