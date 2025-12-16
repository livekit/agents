import pickle
from livekit import durable

@durable.durable
def my_generator():
    for i in range(3):
        yield i

g = my_generator()
print(next(g))  # 0

b = pickle.dumps(g)
g2 = pickle.loads(b)
print(next(g2))  # 1
print(next(g2))  # 2

print(next(g))  # 1
print(next(g))  # 2
