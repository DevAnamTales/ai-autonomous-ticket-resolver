from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/sync-customer-data")
def sync_customer():
    customer_id = request.json.get("customer_id")
    return jsonify({
        "status": "success",
        "message": f"Customer {customer_id} data synced successfully in Sie-CRM/OurTelco."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7002)

