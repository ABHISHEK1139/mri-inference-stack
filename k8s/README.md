# Kubernetes Deployment

## Manifests

| File | Purpose |
| --- | --- |
| `namespace.yaml` | `mri-intelligence` namespace |
| `configmap.yaml` | Runtime environment (VRAM profile, Streamlit flags) |
| `networkpolicy.yaml` | Default-deny plus the egress the app actually needs |
| `deployment.yaml` | Workload, probes, resources, security context |
| `service.yaml` | `ClusterIP` service on port 8501 |
| `pdb.yaml` | Keeps one replica serving during voluntary disruptions |

## Apply in order

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/networkpolicy.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/pdb.yaml
```

## Security posture

- Runs as UID/GID `10001` with `runAsNonRoot` enforced at the pod level.
- `readOnlyRootFilesystem: true`; Streamlit writes to the `/tmp` emptyDir.
- All capabilities dropped, no privilege escalation, `RuntimeDefault` seccomp.
- Default-deny ingress/egress, with explicit allowances for DNS, HTTPS
  egress, and the app port.
- No GPU request by default; uncomment `nvidia.com/gpu` when needed.

## Rollouts

The image tag is versioned (`ghcr.io/ABHISHEK1139/mri-inference-stack:0.2.0`),
not `:latest`, so a rollout is explicit and reversible:

```bash
kubectl -n mri-intelligence set image deployment/mri-intelligence-app \
  mri-app=ghcr.io/ABHISHEK1139/mri-inference-stack:0.2.1
kubectl -n mri-intelligence rollout status deployment/mri-intelligence-app
```

`replicas: 2` with `maxUnavailable: 0` means the service stays up through a
rollout and a node drain.

## Exposing the app

The Service is `ClusterIP`, so the endpoint is not published on node IPs. To
reach it, add an Ingress that terminates TLS and authentication. A commented
example is included in `service.yaml`; create the auth secret first:

```bash
kubectl -n mri-intelligence create secret generic mri-intelligence-auth \
  --from-literal=auth="$(htpasswd -nB admin)"
```

## Model weights

Weights are baked into the image at `/app/weights` by default. To swap models
without rebuilding, uncomment the `weights-volume` mount in `deployment.yaml`
and point it at a `PersistentVolumeClaim` mounted read-only. A startup or
readiness probe failure is the signal that weights did not load.
