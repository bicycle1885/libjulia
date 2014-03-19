# libjulia - a Python binding for libjulia

**Note: This project is highly experimental!**
**Do NOT use this program in your serious product!**

This Python package is a incomplete interface library to Julia.
There are many concerns about object management, but `sample.py` works!


```python
jl_home = os.environ["JULIA_HOME"]
jl.init(jl_home)

## string evaluation
assert jl.eval_string("nothing") is None
assert jl.eval_string("1") is 1
assert jl.eval_string("1.2") == 1.2
assert jl.eval_string("4 * 10") == 40
assert jl.eval_string("sqrt(10)") == math.sqrt(10)

## function calling
f = jl.eval_string("() -> 10")
assert f() == 10
f = jl.eval_string("x -> x * 3.14")
assert f(10) == 10 * 3.14
f = jl.eval_string("(x, y) -> x > y ? true : false")
assert f(2, 1)
assert not f(1, 2)
f = jl.get_base_function("gamma")
assert f(1.0) == math.gamma(1.0)

## module loading
base = jl.load_base_module()
assert base.sin(math.pi / 3) == math.sin(math.pi / 3)
assert base.gamma(1.0) == math.gamma(1.0)
assert base.ifelse(True, 1, 2) == 1
assert base.ifelse(False, 1, 2) == 2
```
