# Ansible Runner

This playbook is intentionally short and practical:

- checks Docker, Docker Compose, and Git LFS
- pulls LFS-tracked model artifacts
- builds and starts the app with Docker Compose

Run it from the repository root:

```powershell
ansible-playbook -i ansible/inventory.ini ansible/site.yml
```

