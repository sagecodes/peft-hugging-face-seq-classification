"""Utility functions for tasks."""


# Convert images to base64 for embedding
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
