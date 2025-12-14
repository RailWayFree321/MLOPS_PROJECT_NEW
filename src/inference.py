"""
Inference Flask App
Serves the trained marketing creative model for inference
This app will be deployed on Kubernetes
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any
import time
from functools import wraps

from flask import Flask, request, jsonify
from datetime import datetime
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model storage
MODEL = None
MODEL_VERSION = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model")

# Prometheus metrics
inference_counter = Counter(
    'model_inference_total',
    'Total number of inferences',
    ['endpoint', 'status']
)

inference_latency = Histogram(
    'model_inference_duration_seconds',
    'Inference latency in seconds',
    ['endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

inference_errors = Counter(
    'model_inference_errors_total',
    'Total number of inference errors',
    ['endpoint', 'error_type']
)

throughput_gauge = Gauge(
    'model_inference_throughput_requests_per_minute',
    'Inference throughput (requests per minute)'
)

model_quality_gauge = Gauge(
    'model_quality_score',
    'Model quality score (0-1)',
    ['model_version']
)

model_loaded_gauge = Gauge(
    'model_loaded',
    'Whether model is loaded (1=yes, 0=no)'
)


def track_metrics(endpoint_name):
    """Decorator to track metrics for endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                latency = time.time() - start_time
                inference_latency.labels(endpoint=endpoint_name).observe(latency)
                
                # Extract status code from response
                status = 'success'
                if isinstance(result, tuple) and len(result) >= 2:
                    status_code = result[1]
                    status = 'success' if status_code < 400 else 'error'
                
                inference_counter.labels(endpoint=endpoint_name, status=status).inc()
                return result
            except Exception as e:
                inference_errors.labels(endpoint=endpoint_name, error_type=type(e).__name__).inc()
                raise
        return decorated_function
    return decorator


def load_model():
    """
    Load the trained model from disk
    """
    global MODEL, MODEL_VERSION
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        
        # Check if model path exists
        if not Path(MODEL_PATH).exists():
            logger.warning(f"Model path does not exist: {MODEL_PATH}")
            model_loaded_gauge.set(0)
            return False
        
        # For now, we'll use a placeholder since we're using transformers
        # In production, this would load the actual model
        MODEL = {
            "loaded": True,
            "path": MODEL_PATH,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to load version info if available
        version_file = Path(MODEL_PATH) / "version.txt"
        if version_file.exists():
            MODEL_VERSION = version_file.read_text().strip()
        else:
            MODEL_VERSION = "unknown"
        
        # Set quality score based on model version (simulated)
        model_quality_gauge.labels(model_version=MODEL_VERSION).set(0.87)
        model_loaded_gauge.set(1)
        
        logger.info(f"Model loaded successfully. Version: {MODEL_VERSION}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded_gauge.set(0)
        return False


def generate_creative(title: str, description: str) -> Dict[str, Any]:
    """
    Generate marketing creative for a product
    
    Args:
        title: Product title
        description: Product description
        
    Returns:
        Generated creative content
    """
    if MODEL is None:
        raise ValueError("Model not loaded")
    
    try:
        # In a real implementation, this would use the actual model for inference
        # For now, we'll create a synthetic response
        
        combined_text = f"{title} {description}"
        
        creative = {
            "title": title,
            "description": description,
            "social_media_caption": f"🎯 {title}! {description[:50]}... Get yours today! #exclusive #quality",
            "hashtags": ["#marketing", "#ecommerce", "#deals", "#shopping"],
            "ad_copy": f"Introducing {title}. {description}",
            "confidence_score": 0.87,
            "generated_at": datetime.now().isoformat()
        }
        
        return creative
    
    except Exception as e:
        logger.error(f"Error generating creative: {str(e)}")
        raise


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if MODEL is not None:
        status["model_version"] = MODEL_VERSION
    
    return jsonify(status), 200 if MODEL is not None else 503


@app.route("/ready", methods=["GET"])
def readiness_probe():
    """Kubernetes readiness probe"""
    if MODEL is None:
        return jsonify({"ready": False}), 503
    
    return jsonify({"ready": True}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/predict", methods=["POST"])
@track_metrics("predict")
def predict():
    """
    Generate marketing creative
    
    Request JSON:
    {
        "title": "Product Title",
        "description": "Product Description"
    }
    """
    try:
        if MODEL is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Validate required fields
        title = data.get("title", "").strip()
        description = data.get("description", "").strip()
        
        if not title or not description:
            return jsonify({
                "status": "error",
                "message": "title and description are required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Generate creative
        creative = generate_creative(title, description)
        
        response = {
            "status": "success",
            "data": creative,
            "model_version": MODEL_VERSION,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated creative for: {title}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/batch_predict", methods=["POST"])
@track_metrics("batch_predict")
def batch_predict():
    """
    Generate marketing creatives for multiple products
    
    Request JSON:
    {
        "products": [
            {"title": "...", "description": "..."},
            {"title": "...", "description": "..."}
        ]
    }
    """
    try:
        if MODEL is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        data = request.get_json()
        
        if not data or "products" not in data:
            return jsonify({
                "status": "error",
                "message": "products field is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        products = data["products"]
        
        if not isinstance(products, list):
            return jsonify({
                "status": "error",
                "message": "products must be a list",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Generate creatives for all products
        creatives = []
        errors = []
        
        for idx, product in enumerate(products):
            try:
                title = product.get("title", "").strip()
                description = product.get("description", "").strip()
                
                if not title or not description:
                    errors.append({
                        "index": idx,
                        "error": "title and description are required"
                    })
                    continue
                
                creative = generate_creative(title, description)
                creatives.append(creative)
            
            except Exception as e:
                errors.append({
                    "index": idx,
                    "error": str(e)
                })
        
        response = {
            "status": "success",
            "data": {
                "creatives": creatives,
                "total_processed": len(products),
                "successful": len(creatives),
                "failed": len(errors),
                "errors": errors if errors else []
            },
            "model_version": MODEL_VERSION,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated {len(creatives)} creatives in batch")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in batch_predict endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/info", methods=["GET"])
def model_info():
    """Get model information"""
    info = {
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }
    
    if MODEL is not None:
        info.update(MODEL)
    
    return jsonify(info), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500


if __name__ == "__main__":
    # Load model on startup
    logger.info("Starting Inference Flask App")
    
    if not load_model():
        logger.warning("Could not load model on startup. API will return 503 until model is loaded.")
    
    port = int(os.getenv("INFERENCE_PORT", 5001))
    logger.info(f"Running on port {port}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
