from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/retry-order")
def retry_order():
    order_id = request.json.get("order_id")
    return jsonify({
        "status": "success",
        "message": f"Order {order_id} has been successfully retried in OSM."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001)

