ML project for Masters Work (University of Latvia)

VRP problem instance (edges reduced to k-nearest) is passed to the ML, which returns "Optimality score" for each edge, ranging from 0 to 1:
- 0: edge is not optimal;
- 1: edge is optimal;

Domain: VRP with multiple couriers
- List<Courier>: each courier has depot + own route with clients;
- List<Client>: each client must be visited + has own location in (x, y) format

built on Python/PyTorch/PyVRP
