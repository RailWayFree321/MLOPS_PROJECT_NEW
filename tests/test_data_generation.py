"""
Tests for data generation module
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from src.data_generation import generate_product_data, app


class TestDataGeneration:
    """Test data generation functions"""
    
    def test_generate_product_data_count(self):
        """Test that correct number of products is generated"""
        num_records = 50
        products = generate_product_data(num_records)
        
        assert len(products) == num_records
        assert isinstance(products, list)
    
    def test_generate_product_data_structure(self):
        """Test that generated products have correct structure"""
        products = generate_product_data(5)
        
        required_fields = {"id", "title", "category", "description", "price", "rating", "created_at"}
        
        for product in products:
            assert isinstance(product, dict)
            assert required_fields.issubset(set(product.keys()))
            
            # Validate field types
            assert isinstance(product["id"], int)
            assert isinstance(product["title"], str)
            assert isinstance(product["category"], str)
            assert isinstance(product["description"], str)
            assert isinstance(product["price"], float)
            assert isinstance(product["rating"], (int, float))
            assert isinstance(product["created_at"], str)
    
    def test_generate_product_data_valid_ranges(self):
        """Test that generated data is within valid ranges"""
        products = generate_product_data(20)
        
        for product in products:
            # Price should be between 10 and 500
            assert 10 <= product["price"] <= 500
            
            # Rating should be between 2 and 5
            assert 2 <= product["rating"] <= 5
            
            # Validate non-empty strings
            assert len(product["title"]) > 0
            assert len(product["category"]) > 0
            assert len(product["description"]) > 0
    
    def test_generate_product_data_unique_ids(self):
        """Test that generated products have unique IDs"""
        products = generate_product_data(30)
        ids = [p["id"] for p in products]
        
        assert len(ids) == len(set(ids)), "Product IDs should be unique"
    
    def test_product_categories_valid(self):
        """Test that product categories are from predefined list"""
        from src.data_generation import CATEGORIES
        
        products = generate_product_data(50)
        
        for product in products:
            assert product["category"] in CATEGORIES


class TestFlaskApp:
    """Test Flask application endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_generate_data_endpoint(self, client):
        """Test data generation endpoint"""
        response = client.get("/generate_data")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert data["status"] == "success"
        assert "message" in data
        assert "count" in data
        assert "data" in data
        assert "timestamp" in data
        
        # Check data count
        assert data["count"] == 50
        assert len(data["data"]) == 50
    
    def test_generate_data_endpoint_structure(self, client):
        """Test that endpoint returns correctly structured data"""
        response = client.get("/generate_data")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check each product in response
        for product in data["data"]:
            assert "id" in product
            assert "title" in product
            assert "category" in product
            assert "description" in product
            assert "price" in product
            assert "rating" in product
            assert "created_at" in product
    
    def test_generate_data_endpoint_consistency(self, client):
        """Test that endpoint can be called multiple times"""
        response1 = client.get("/generate_data")
        response2 = client.get("/generate_data")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        
        assert data1["count"] == data2["count"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
