ML project for Masters Work (University of Latvia)

VRP problem instance (edges reduced to k-nearest) is passed to the ML, which returns "Optimality score" for each edge, ranging from 0 to 1:
- 0: edge is not optimal;
- 1: edge is optimal;

Domain: VRP with multiple couriers
- List<Courier>: each courier has depot + own route with clients;
- List<Client>: each client must be visited + has own location in (x, y) format

built on Python/PyTorch/PyVRP

Setup:
- Download ML model: https://www.dropbox.com/scl/fi/bb4zxhcfhk7e0fr7fi1pr/best.pt?rlkey=xstk4a6bg60heviwy2wvsbr9a&st=3pqmt7nw&dl=0
- create folder runs/edge_gnn
- save model as "best.pt" in this folder
- run server\run_server.py file

Usage:
- Send VRP instance (encoded in the GnnInput format) to the POST http://localhost:8000/analyze API
- get instance edge scores (encoded in the GnnOutput format)

Or use https://github.com/sakuraaah/CombOpt-Masters project (where this API is already integrated)
