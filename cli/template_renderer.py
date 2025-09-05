"""
Simple template renderer for the HTTP server
Will be replaced with Jinja2 in the FastAPI phase
"""

import os
import re

class TemplateRenderer:
    def __init__(self, template_dir="templates"):
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_dir = os.path.join(current_dir, template_dir)
    
    def render(self, template_name, context=None):
        """Render a template with the given context"""
        if context is None:
            context = {}
        
        template_path = os.path.join(self.template_dir, template_name)
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return f"Template {template_name} not found at {template_path}"
        
        # Handle {% block %} and {% extends %} first
        content = self._process_blocks(content, context)
        
        # Simple template variable replacement
        # Replace {{ variable }} with context value
        for key, value in context.items():
            placeholder = f"{{{{ {key} }}}}"
            content = content.replace(placeholder, str(value))
        
        # Process Jinja2-style conditionals and loops
        content = self._process_conditionals(content, context)
        
        return content
    
    def _process_blocks(self, content, context):
        """Process template blocks and inheritance"""
        # Handle {% extends "base.html" %}
        extends_match = re.search(r'{%\s*extends\s*["\']([^"\']+)["\']\s*%}', content)
        if extends_match:
            base_template = extends_match.group(1)
            base_content = self.render(base_template, context)
            
            # Extract blocks from content
            blocks = {}
            block_pattern = r'{%\s*block\s+(\w+)\s*%}(.*?){%\s*endblock\s*%}'
            for match in re.finditer(block_pattern, content, re.DOTALL):
                block_name = match.group(1)
                block_content = match.group(2).strip()
                blocks[block_name] = block_content
            
            # Replace blocks in base template
            for block_name, block_content in blocks.items():
                block_placeholder = f"{{{{ {block_name} }}}}"
                base_content = base_content.replace(block_placeholder, block_content)
            
            return base_content
        
        # If no extends, process blocks in place
        # Replace {% block name %}content{% endblock %} with just the content
        block_pattern = r'{%\s*block\s+\w+\s*%}(.*?){%\s*endblock\s*%}'
        content = re.sub(block_pattern, r'\1', content, flags=re.DOTALL)
        
        return content
    
    def _process_conditionals(self, content, context):
        """Process Jinja2-style conditionals"""
        # Handle {% if active_page == 'dashboard' %}active{% endif %}
        conditional_pattern = r'{%\s*if\s+(\w+)\s*==\s*["\']([^"\']+)["\']\s*%}(.*?){%\s*endif\s*%}'
        
        def replace_conditional(match):
            var_name = match.group(1)
            expected_value = match.group(2)
            conditional_content = match.group(3)
            
            if context.get(var_name) == expected_value:
                return conditional_content
            else:
                return ''
        
        content = re.sub(conditional_pattern, replace_conditional, content, flags=re.DOTALL)
        
        # Handle simple {% if %} statements
        simple_if_pattern = r'{%\s*if\s+(\w+)\s*%}(.*?){%\s*endif\s*%}'
        
        def replace_simple_if(match):
            var_name = match.group(1)
            conditional_content = match.group(2)
            
            if context.get(var_name):
                return conditional_content
            else:
                return ''
        
        content = re.sub(simple_if_pattern, replace_simple_if, content, flags=re.DOTALL)
        
        return content
    
    def render_dashboard(self, context=None):
        """Render the main dashboard template"""
        context = context or {}
        context['active_page'] = 'dashboard'
        return self.render("dashboard.html", context)
    
    def render_base(self, context=None):
        """Render the base template"""
        return self.render("base.html", context or {})
