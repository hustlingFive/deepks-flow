Build base_dflow_deepks
```
docker build --platform linux/amd64 -f ./Dockerfile.base -t hustling/base-dflow-deepks:1.0 .
```

Build ABACUS with gnu
This dockerfile comes from:
https://github.com/deepmodeling/abacus-develop/blob/develop/Dockerfile.gnu
```
docker build --platform linux/amd64 -f ./Dockerfile.gnu -t hustling/abacus-gnu:1.0 .
```
**this image currently cant use with this workflow**