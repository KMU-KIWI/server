import numpy as np
import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "GENERATED_SPEECH"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        print("Initialized...")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            # Create an InferenceRequest object. `model_name`,
            # `requested_output_names`, and `inputs` are the required arguments and
            # must be provided when constructing an InferenceRequest object. Make sure
            # to replace `inputs` argument with a list of `pb_utils.Tensor` objects.
            input0 = pb_utils.get_input_tensor_by_name(request, "RAW_SPEECH")

            output0 = self.execute_asr(input0)

            # Decide the next steps for model execution based on the received output
            # tensors. It is possible to use the same output tensors to for the final
            # inference response too.
            np_output0 = output0.as_numpy()
            if np_output0.item() == "":
                # text was not recognized from audio
                input0 = pb_utils.Tensor("GENERATED_SPEECH", np.array([0.0]))

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output0]
                )
                responses.append(inference_response)
            else:
                input0 = pb_utils.Tensor("QUESTION", np_output0)

                output0 = self.execute_nlp(input0)

                input0 = pb_utils.Tensor("RAW_TEXT", output0.as_numpy())
                output0 = self.execute_tts(input0)

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output0]
                )
                responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def execute_asr(self, *inputs):
        inference_request = pb_utils.InferenceRequest(
            model_name="asr",
            requested_output_names=["RECOGNIZED_TEXT"],
            inputs=[*inputs],
        )

        inference_response = inference_request.exec()

        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output0 = pb_utils.get_output_tensor_by_name(
                inference_response, "RECOGNIZED_TEXT"
            )

            return output0

    def execute_nlp(self, *inputs):
        inference_request = pb_utils.InferenceRequest(
            model_name="nlp",
            requested_output_names=["ANSWER"],
            inputs=[*inputs],
        )

        inference_response = inference_request.exec()

        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output0 = pb_utils.get_output_tensor_by_name(
                inference_response,
                "ANSWER",
            )

            return output0

    def execute_tts(self, *inputs):
        inference_request = pb_utils.InferenceRequest(
            model_name="tts",
            requested_output_names=["GENERATED_SPEECH"],
            inputs=[*inputs],
        )

        inference_response = inference_request.exec()

        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output0 = pb_utils.get_output_tensor_by_name(
                inference_response,
                "GENERATED_SPEECH",
            )

            return output0

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
