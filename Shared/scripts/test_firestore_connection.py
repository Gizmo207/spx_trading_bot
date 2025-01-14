from google.cloud import firestore

# Initialize Firestore
db = firestore.Client()

# Function to fetch sample data
def fetch_sample():
    collection_ref = db.collection("historical_data")  # Replace with your actual collection name
    docs = collection_ref.limit(5).stream()
    for doc in docs:
        print(f"{doc.id} => {doc.to_dict()}")

# Run the function
fetch_sample()
