from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/fix-asset")
def fix_asset():
    asset_id = request.json.get("asset_id")
    return jsonify({
        "status": "success",
        "message": f"Asset mismatch fixed for asset {asset_id} in BRM."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7003)

