"""
VULNERABLE WEB APPLICATION - FOR TESTING ONLY
Contains Critical, High, Medium, and Low severity issues
"""

import os
import pickle
import subprocess
import tempfile
import hashlib
import random
from datetime import datetime
from flask import Flask, request, render_template_string, session, redirect
import sqlite3
import xml.etree.ElementTree as ET
import yaml

app = Flask(__name__)

# ============================================
# CRITICAL SEVERITY ISSUES
# ============================================

# CRITICAL: Hardcoded credentials
DATABASE_PASSWORD = "admin123"
API_KEY = "sk_live_51234567890abcdef"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
MASTER_PASSWORD = "SuperSecret2024!"

# CRITICAL: SQL Injection vulnerability
def get_user_data(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Direct string interpolation - SQL injection risk
    query = f"SELECT * FROM users WHERE username='{username}'"
    cursor.execute(query)
    return cursor.fetchall()

# CRITICAL: Remote Code Execution via deserialization
def load_user_session(session_data):
    # Unsafe deserialization of untrusted data
    return pickle.loads(session_data)

# CRITICAL: Command Injection
def process_file(filename):
    # Direct user input in shell command
    os.system(f"cat /uploads/{filename}")
    subprocess.call(f"convert {filename} -resize 800x600 output.jpg", shell=True)

# CRITICAL: eval() with user input
def calculate_expression(expr):
    # Arbitrary code execution
    result = eval(expr)
    return result

# CRITICAL: Insecure cryptography - hardcoded key
ENCRYPTION_KEY = b'1234567890123456'  # 16 bytes for AES
SECRET_SALT = "fixed_salt_123"

# ============================================
# HIGH SEVERITY ISSUES
# ============================================

# HIGH: Broken authentication
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Password stored in plain text
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = cursor.execute(query).fetchone()
    if result:
        # Predictable session token
        session['token'] = hashlib.md5(username.encode()).hexdigest()
        session['admin'] = True  # No role verification
        return True
    return False

# HIGH: Path Traversal vulnerability
def read_file(filename):
    # No path sanitization
    with open(f"/var/www/uploads/{filename}") as f:
        return f.read()

# HIGH: SSRF (Server-Side Request Forgery)
def fetch_url(url):
    import urllib.request
    # No URL validation - can access internal services
    response = urllib.request.urlopen(url)
    return response.read()

# HIGH: XXE (XML External Entity) vulnerability
def parse_xml(xml_data):
    # XML parser not configured to prevent XXE
    tree = ET.fromstring(xml_data)
    return tree

# HIGH: Weak password hashing
def hash_password(password):
    # MD5 is cryptographically broken
    return hashlib.md5(password.encode()).hexdigest()

# HIGH: Insecure random for security purposes
def generate_reset_token():
    # random is not cryptographically secure
    return str(random.randint(100000, 999999))

# HIGH: YAML deserialization vulnerability
def load_config(config_data):
    # yaml.load() allows arbitrary code execution
    return yaml.load(config_data)

# HIGH: No rate limiting on sensitive operations
def reset_password(email):
    token = generate_reset_token()
    # No rate limiting - brute force possible
    send_email(email, token)

# ============================================
# MEDIUM SEVERITY ISSUES
# ============================================

# MEDIUM: XSS (Cross-Site Scripting) vulnerability
@app.route('/search')
def search():
    query = request.args.get('q', '')
    # Unescaped user input rendered in HTML
    return render_template_string(f"<h1>Search results for: {query}</h1>")

# MEDIUM: CSRF (Cross-Site Request Forgery) - no token
@app.route('/transfer', methods=['POST'])
def transfer_money():
    # No CSRF token verification
    amount = request.form.get('amount')
    to_account = request.form.get('to')
    process_transfer(to_account, amount)
    return "Transfer complete"

# MEDIUM: Information disclosure through error messages
def get_user(user_id):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
        return cursor.fetchone()
    except Exception as e:
        # Detailed error messages expose system information
        return f"Database error: {str(e)}, Connection: {conn}, User: {os.getenv('USER')}"

# MEDIUM: Insecure direct object reference
@app.route('/document/<doc_id>')
def get_document(doc_id):
    # No authorization check
    with open(f"/documents/{doc_id}.pdf", 'rb') as f:
        return f.read()

# MEDIUM: Weak session management
app.secret_key = "123"  # Weak and predictable secret key

# MEDIUM: Missing security headers
@app.route('/')
def home():
    # No CSP, X-Frame-Options, etc.
    return "<h1>Welcome</h1>"

# MEDIUM: Insecure file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    # No file type validation
    # No size limit
    file.save(f"/uploads/{file.filename}")
    return "File uploaded"

# MEDIUM: Open redirect vulnerability
@app.route('/redirect')
def redirect_user():
    url = request.args.get('url')
    # No URL validation
    return redirect(url)

# ============================================
# LOW SEVERITY ISSUES
# ============================================

# LOW: Race condition
user_balance = 0
def withdraw(amount):
    global user_balance
    if user_balance >= amount:
        # Non-atomic operation
        user_balance -= amount
        return True
    return False

# LOW: Resource leak - file not closed
def process_log_file(filename):
    log = open(filename)
    data = log.read()
    # File never closed
    return data

# LOW: Use of deprecated/insecure function
def create_temp_file():
    # mktemp() is insecure - race condition
    return tempfile.mktemp()

# LOW: Missing input validation
def set_age(age):
    # No validation - negative or huge values possible
    user_age = age
    return user_age

# LOW: Logging sensitive data
def login_attempt(username, password):
    # Password logged in plain text
    print(f"Login attempt: {username}:{password}")
    logger.info(f"User {username} with password {password} attempted login")

# LOW: High cyclomatic complexity
def complex_validation(a, b, c, d, e, f, g):
    if a > 0:
        if b < 10:
            if c == "test":
                if d and e:
                    if f > g:
                        if a + b > 20:
                            if c in ["test", "prod"]:
                                if len(d) > 5:
                                    return True
    return False

# LOW: Code duplication
def save_user(name, email):
    user = User()
    user.name = name
    user.email = email
    user.created = datetime.now()
    user.status = "active"
    db.save(user)
    return user

def save_post(title, content):
    post = Post()
    post.title = title
    post.content = content
    post.created = datetime.now()
    post.status = "active"
    db.save(post)
    return post

def save_comment(text, author):
    comment = Comment()
    comment.text = text
    comment.author = author
    comment.created = datetime.now()
    comment.status = "active"
    db.save(comment)
    return comment

# LOW: Magic numbers
def calculate_price(quantity):
    if quantity > 100:
        return quantity * 9.99 * 0.85
    elif quantity > 50:
        return quantity * 9.99 * 0.90
    else:
        return quantity * 9.99

# LOW: Commented out code
def process_payment(amount):
    # old_process_payment(amount)
    # calculate_tax(amount)
    # verify_funds(amount)
    new_payment_processor(amount)

# LOW: TODO comments indicating unfinished security features
def verify_permissions(user, resource):
    # TODO: Implement proper RBAC
    # TODO: Add permission caching
    # TODO: Check resource ownership
    return True

# ============================================
# ADDITIONAL VULNERABILITIES
# ============================================

# Unvalidated redirect and forward
@app.route('/goto')
def goto():
    target = request.args.get('target')
    return redirect(target)

# Mass assignment vulnerability
@app.route('/update_profile', methods=['POST'])
def update_profile():
    user = get_current_user()
    # Allows setting any attribute including 'is_admin'
    for key, value in request.form.items():
        setattr(user, key, value)
    user.save()
    return "Profile updated"

# Insufficient logging
def delete_user(user_id):
    # Critical operation with no logging
    db.execute(f"DELETE FROM users WHERE id={user_id}")

# Debug mode in production
app.debug = True
app.config['TESTING'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True

# Insecure cookie configuration
@app.route('/set_cookie')
def set_cookie():
    resp = make_response("Cookie set")
    # No HttpOnly, Secure, or SameSite flags
    resp.set_cookie('session_id', generate_token())
    return resp

# N+1 query problem
def get_users_with_posts():
    users = db.query("SELECT * FROM users")
    for user in users:
        # Separate query for each user
        posts = db.query(f"SELECT * FROM posts WHERE user_id={user.id}")
        user.posts = posts
    return users

# Timing attack vulnerability
def verify_api_key(provided_key):
    actual_key = get_api_key_from_db()
    # String comparison vulnerable to timing attacks
    if provided_key == actual_key:
        return True
    return False

# Use of assert for security checks
def admin_only_function(user):
    # assert removed in optimized Python
    assert user.is_admin, "Not authorized"
    perform_admin_action()

# Memory leak - circular reference
class Node:
    def __init__(self):
        self.ref = self

# Uncontrolled resource consumption
@app.route('/search_all')
def search_all():
    query = request.args.get('q')
    # No pagination - can return millions of records
    return db.query(f"SELECT * FROM products WHERE name LIKE '%{query}%'")

if __name__ == '__main__':
    # Running with 0.0.0.0 exposes to all interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
