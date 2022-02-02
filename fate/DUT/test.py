import simpy

def deli2(env, resource, id, latency):
    print("Starting ", id)
    # Grab the resource
    with resource.request() as req:
        print("I am waiting {0} at time {1} with request {2}".format(id, env.now, req))
        yield req

        print("I am {0} going to tick at time {1}".format(id, env.now))
        yield env.timeout(latency)
        print("I am {0} releasing at time {1}".format(id, env.now))

def deli(env, resource, id, latency):
    print("Starting ", id)
    # Grab the resource
    with resource.request() as req:
        print("I am waiting {0} at time {1} with request {2}".format(id, env.now, req))
        yield req

        print("I am {0} going to tick at time {1}".format(id, env.now))
        yield env.timeout(latency)
        print("I am {0} releasing at time {1}".format(id, env.now))

if __name__ == "__main__":
    env = simpy.Environment()
    resource = simpy.Resource(env, 3)
    for i in range(4):
        env.process(deli(env, resource, i, 10))
    for i in range(4):
        env.process(deli2(env, resource, 4+i, 10))
    env.run(until=50)

