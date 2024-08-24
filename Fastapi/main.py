from fastapi import FastAPI
app = FastAPI()
#my_first_api = FastAPI()
#path
#POST, PUT, DELETE, GET

@app.get("/")#Path operation decorator
async def root():
    return {"Message" : "Hello world from FastAPI"}

@app.get("/demo")
def demo_func():
    return {"message": "This is output from demo function"}

@app.post("/post_demo")
def demo_func():
    return {"message": "This is output from post demo function"}

""""
Post: to create data
Get : to read data
Put : to update data
Delete : to delete data
"""