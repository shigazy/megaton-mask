import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from '@/contexts/AuthContext';
import { StatusProvider } from '@/contexts/StatusContext';
import { Header } from '@/components/Header';
import { Snackbar } from '@/components/Snackbar';
import { CreditsProvider } from '@/contexts/CreditsContext';

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Megaton Roto",
  description: "The most advanced rotoscoper",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AuthProvider>
          <StatusProvider>
            <CreditsProvider>
              <div className="min-h-screen flex flex-col">
                <Snackbar />
                <main className="flex-1">
                  {children}
                </main>
              </div>
            </CreditsProvider>
          </StatusProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
