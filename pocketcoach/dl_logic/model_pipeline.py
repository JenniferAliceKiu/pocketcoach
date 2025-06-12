from transformers import Pipeline
import tensorflow as tf
from pocketcoach.dl_logic.data import get_data, clean_data_set, pad, clean

class ModelPipeline(Pipeline):

    def __init__(self, model, tokenizer):
        # Do NOT call super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        # Required internal attributes to mimic Hugging Face pipeline behavior
        self._batch_size = 1
        self._num_workers = 0
        self.device = -1
        self.framework = "tf"
        self.model_input_names = []
        self._preprocess_params = {}
        self._forward_params = {}
        self._postprocess_params = {}

        # 👇 The missing attribute causing your latest error
        self.call_count = 0

    def _sanitize_parameters(self, **kwargs):
        # No extra params used here
        return {}, {}, {}

    def preprocess(self, inputs):
        # Expect `inputs` as a string or dict
        # Tokenization/preprocessing should match your model's training
        # Example: split on spaces and convert to indices…
        cleaned_text = clean(inputs)
        padded_input = pad([cleaned_text], self.tokenizer)

        return {"input": tf.convert_to_tensor(padded_input, dtype=tf.int32)}

    def _forward(self, model_inputs):
        input_tensor = model_inputs["input"]
        output = self.model(input_tensor, training=False)
        return {"output": output}

    def postprocess(self, model_outputs):
        # Process the raw model output into something user-friendly
        # Example: softmax and return labels and scores
        output_tensor = model_outputs["output"]
        probs = tf.nn.softmax(output_tensor).numpy().tolist()
        return [{"score": s, "label": idx} for idx, s in enumerate(probs[0])]
