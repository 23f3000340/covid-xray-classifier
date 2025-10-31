import threading

# 1. Race condition - global without lock
user_count = 0

def add_user():
    global user_count
    user_count += 1  # Non-atomic

# 2. Memory leak - file not closed
def process_log(filename):
    log = open(filename)
    data = log.read()
    return data

# 3. Auth issue - no permission check
def admin_delete(user_id):
    db.execute(f"DELETE FROM users WHERE id={user_id}")

# 4. N+1 query problem
def list_with_details():
    items = db.query("SELECT * FROM items")
    for item in items:
        details = db.query(f"SELECT * FROM details WHERE item_id={item.id}")

# 5. High complexity
def validate(x, y, z, a, b):
    if x > 0:
        if y < 10:
            if z == "test":
                if a and b:
                    if x + y > z:
                        return True
    return False

# 6. Code duplication
def save_user(name):
    user = User()
    user.name = name
    user.created = datetime.now()
    db.save(user)
    return user

def save_post(title):
    post = Post()
    post.title = title
    post.created = datetime.now()
    db.save(post)
    return post
