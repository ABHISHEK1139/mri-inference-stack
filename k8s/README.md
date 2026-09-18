# Kubernetes Deployment (Minimal)

This manifests set is intentionally compact:

- `namespace.yaml`
- `configmap.yaml`
- `deployment.yaml`
- `service.yaml`

The deployment now includes:

- startup, readiness, and liveness probes against the Streamlit health endpoint
- explicit CPU and memory requests/limits for reproducible scheduling
- a conservative rolling update strategy for safer rollouts

Apply:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

By default, the deployment uses the model weights baked directly into the container image (`/app/weights`). The `outputs-volume` preserves runtime exports.

If you prefer external weight provisioning (e.g., via a PersistentVolumeClaim or NFS), uncomment the `weights-volume` mount in `k8s/deployment.yaml`.

