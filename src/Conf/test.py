def msp(logits, **kwargs):
    # msp is defined to only accept 'logits'
    return -max(logits)

def odin(logits, temperature, **kwargs):
    print("pdin arg", logits, temperature)
    # odin requires a 'temperature' argument.
    return -max([x / temperature for x in logits])

# Demonstrate calling directly:
logits = [1, 2, 3]

# This works:
print("odin without wrapper:", odin(logits, temperature=1.0))

# This will raise an error because msp doesn't accept a 'temperature' parameter.
# try:
#     print("msp without wrapper:", msp(logits, temperature=1.0))
# except TypeError as e:
#     print("Error calling msp without wrapper:", e)

# Now, define a simple wrapper that collects extra parameters:
class Wrapper:
    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *call_args, **call_kwargs):
        # Merge the stored kwargs with call-time kwargs.
        return self.method(*self.args, *call_args, **self.kwargs, **call_kwargs)

# Wrap the functions.

config = {"temperature":2.0,
"dummy":3.
#  "dummy":0
 }

print("test ", odin(logits, **config))
# try:
#     print(" msp(logits, **config ", msp(logits, **config))
# except TypeError as e:
#     print("Error calling msp wwith *kwargs", e)
wrapped_msp = Wrapper(msp, temperature=1.0)  # temperature will be ignored by msp.
wrapped_odin = Wrapper(odin, **config)

# # Now, calling the wrapped functions:
# print("wrapped_msp:", wrapped_msp(logits))     # Works, extra parameter is provided but msp ignores it.
# print("wrapped_odin:", wrapped_odin(logits))     # Works as expected.
