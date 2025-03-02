import torch.quantization

class QuantizedInferenceManager:
    def __init__(self):
        self.model_path = "sam2/checkpoints/sam2.1_hiera_large.pt"
        self.predictor = self._load_quantized_model()
        
    def _load_quantized_model(self):
        # Load the original model
        model = get_predictor(self.model_path)
        
        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate the model (you'd need a calibration dataset)
        # self._calibrate_model(model)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model 