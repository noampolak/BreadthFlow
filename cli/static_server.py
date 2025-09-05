"""
Simple static file server for CSS and JavaScript files
Will be replaced with proper static file serving in FastAPI
"""

import os
import mimetypes

class StaticFileServer:
    def __init__(self, static_dir="static"):
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(current_dir, static_dir)
        # Initialize mimetypes
        mimetypes.init()
    
    def serve_file(self, path):
        """Serve a static file and return (content, content_type, status_code)"""
        # Remove leading slash and static prefix
        if path.startswith('/static/'):
            path = path[8:]  # Remove '/static/'
        
        file_path = os.path.join(self.static_dir, path)
        
        # Security check - prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(self.static_dir)):
            return None, None, 403
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                if file_path.endswith('.js'):
                    content_type = 'application/javascript'
                elif file_path.endswith('.css'):
                    content_type = 'text/css'
                else:
                    content_type = 'application/octet-stream'
            
            return content, content_type, 200
            
        except FileNotFoundError:
            return None, None, 404
        except Exception as e:
            return str(e).encode(), 'text/plain', 500
    
    def is_static_request(self, path):
        """Check if the request is for a static file"""
        return path.startswith('/static/')
