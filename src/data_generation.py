"""
Data Generation Flask App
Generates random e-commerce product data for model training
"""
import random
import json
from flask import Flask, jsonify
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Sample product categories and adjectives for variety
CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Sports",
    "Beauty", "Toys", "Books", "Health"
]

ADJECTIVES = [
    "Premium", "Professional", "Portable", "Lightweight",
    "Durable", "Eco-friendly", "High-performance", "Stylish",
    "Comfortable", "Waterproof", "Smart", "Fast"
]

PRODUCT_TYPES = [
    "Headphones", "Smartphone", "Laptop", "T-Shirt", "Running Shoes",
    "Coffee Maker", "Backpack", "Watch", "Camera", "Speaker",
    "Jacket", "Tablet", "Keyboard", "Mouse", "Monitor",
    "Desk Lamp", "USB Cable", "Phone Stand", "Microphone", "Webcam"
]


def generate_product_data(num_records: int = 50) -> list:
    """
    Generate random product data for training
    
    Args:
        num_records: Number of product records to generate
        
    Returns:
        List of product dictionaries
    """
    products = []
    
    for i in range(num_records):
        product_type = random.choice(PRODUCT_TYPES)
        adjective = random.choice(ADJECTIVES)
        category = random.choice(CATEGORIES)
        price = round(random.uniform(10, 500), 2)
        rating = round(random.uniform(2, 5), 1)
        
        product = {
            "id": i + 1,
            "title": f"{adjective} {product_type}",
            "category": category,
            "description": f"High-quality {product_type.lower()} perfect for everyday use. "
                         f"Features premium materials and advanced technology. "
                         f"Rated {rating}/5 by customers.",
            "price": price,
            "rating": rating,
            "created_at": datetime.now().isoformat()
        }
        products.append(product)
    
    logger.info(f"Generated {num_records} product records")
    return products


@app.route("/generate_data", methods=["GET"])
def generate_data():
    """
    API endpoint to generate product data
    
    Returns:
        JSON response with generated product data
    """
    try:
        num_records = 50
        products = generate_product_data(num_records)
        
        response = {
            "status": "success",
            "message": f"Generated {num_records} product records",
            "count": len(products),
            "data": products,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data generation endpoint called - returned {len(products)} records")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200


if __name__ == "__main__":
    logger.info("Starting Data Generation Flask App on port 5000")
    app.run(host="0.0.0.0", port=5002, debug=True)
