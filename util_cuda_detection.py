import torch
from transformers import AutoTokenizer, AutoModel


def bytes_to_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 2)


def detect_cuda_support() -> None:
    print("=== Détection CUDA (via PyTorch) ===")
    print(f"Version de PyTorch           : {torch.__version__}")
    print(f"Version CUDA compilée (torch): {torch.version.cuda}")

    is_available = torch.cuda.is_available()
    print(f"CUDA disponible               : {'Oui' if is_available else 'Non'}")

    if not is_available:
        return

    device_count = torch.cuda.device_count()
    print(f"Nombre de GPU détectés        : {device_count}")

    for device_index in range(device_count):
        name = torch.cuda.get_device_name(device_index)
        capability_major, capability_minor = torch.cuda.get_device_capability(device_index)
        props = torch.cuda.get_device_properties(device_index)
        total_mem_gib = bytes_to_gib(props.total_memory)

        print(f"- GPU {device_index} : {name}")
        print(f"  Capability SM             : {capability_major}.{capability_minor}")
        print(f"  Mémoire totale            : {total_mem_gib} GiB")

    print(f"cuDNN activé                 : {'Oui' if torch.backends.cudnn.enabled else 'Non'}")


if __name__ == "__main__":
    detect_cuda_support()


