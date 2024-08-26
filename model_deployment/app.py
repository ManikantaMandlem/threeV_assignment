from flask import Flask, request, jsonify
import sys
import argparse
import os

# sys.path.append("/Users/manikantamandlem/Desktop/threeV_assignment/model_deployment/segmentation_model")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "segmentation_model"))
)
from inf_helpers import InfObject


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-weights",
        "-w",
        type=str,
        required=False,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "artifacts", "swin_Unet_v0.0.1.pth")
        ),
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        required=False,
        default=5050,
        help="Port to expose the endpoint",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    app = Flask(__name__)
    app.config["inf_obj"] = InfObject(model_weights=args.model_weights)

    @app.route("/seg_mask", methods=["POST"])
    def get_mask():
        data = request.json
        try:
            app.config["inf_obj"].load_image(data["image_path"])
            app.config["inf_obj"].save_result()
            response = {"status": "success", "result": "Mask displayed"}
            return jsonify(response)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    app.run(host="0.0.0.0", port=args.port)
