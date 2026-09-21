import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# ==========================================================
# IMPORT FROM YOUR EXISTING FILE
# ==========================================================

from adversarial_face_eval import (
    build_models,
    load_image,
    denorm,
    fgsm_attack,
    ifgsm_attack,
    bpda_attack
)

# ==========================================================
# SETTINGS
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# SAVE IMAGE
# ==========================================================

def save_tensor_image(tensor, filename):

    img = denorm(tensor)

    img = img.squeeze(0)

    img = img.clamp(0, 1)

    img = transforms.ToPILImage()(img.cpu())

    img.save(filename)


# ==========================================================
# SAVE PERTURBATION IMAGE
# ==========================================================

def save_noise_visualization(clean, adv, filename):

    noise = (adv - clean).abs()

    noise = noise.squeeze(0).detach().cpu().numpy()

    noise = np.transpose(noise, (1, 2, 0))

    noise = noise / (noise.max() + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(noise)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()


# ==========================================================
# GET EMBEDDING
# ==========================================================

def get_embedding(model, image_tensor):

    with torch.no_grad():
        emb = model.embed_tensor(image_tensor)

    return emb


# ==========================================================
# SAVE COMPARISON FIGURE
# ==========================================================

def save_comparison(clean, fgsm, ifgsm, bpda, filename):

    imgs = [clean, fgsm, ifgsm, bpda]

    titles = [
        "Clean",
        "FGSM",
        "I-FGSM",
        "BPDA"
    ]

    plt.figure(figsize=(16, 4))

    for i, img in enumerate(imgs):

        img = denorm(img)

        img = img.squeeze(0)

        img = img.clamp(0, 1)

        img = img.permute(1, 2, 0)

        plt.subplot(1, 4, i + 1)

        plt.imshow(img.cpu())

        plt.title(titles[i])

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()


# ==========================================================
# MAIN EVALUATION
# ==========================================================

def evaluate_image(
        model,
        image_path,
        reference_path,
        output_dir="attack_results"):

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading images...")

    x = load_image(image_path)

    x_ref = load_image(reference_path)

    clean_emb = get_embedding(model, x)

    ref_emb = get_embedding(model, x_ref)

    clean_similarity = F.cosine_similarity(
        clean_emb,
        ref_emb
    ).item()

    print(f"Original Similarity = {clean_similarity:.4f}")

    # --------------------------------------------------
    # ATTACKS
    # --------------------------------------------------

    fgsm_img = fgsm_attack(
        model,
        x,
        x_ref,
        epsilon=0.10,
        label=1
    )

    ifgsm_img = ifgsm_attack(
        model,
        x,
        x_ref,
        epsilon=0.10,
        label=1
    )

    bpda_img = bpda_attack(
        model,
        x,
        x_ref,
        epsilon=0.10,
        label=1
    )

    # --------------------------------------------------
    # SAVE IMAGES
    # --------------------------------------------------

    save_tensor_image(
        x,
        f"{output_dir}/clean.png"
    )

    save_tensor_image(
        fgsm_img,
        f"{output_dir}/fgsm.png"
    )

    save_tensor_image(
        ifgsm_img,
        f"{output_dir}/ifgsm.png"
    )

    save_tensor_image(
        bpda_img,
        f"{output_dir}/bpda.png"
    )

    # --------------------------------------------------
    # SAVE NOISE
    # --------------------------------------------------

    save_noise_visualization(
        x,
        fgsm_img,
        f"{output_dir}/fgsm_noise.png"
    )

    save_noise_visualization(
        x,
        ifgsm_img,
        f"{output_dir}/ifgsm_noise.png"
    )

    save_noise_visualization(
        x,
        bpda_img,
        f"{output_dir}/bpda_noise.png"
    )

    # --------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------

    save_comparison(
        x,
        fgsm_img,
        ifgsm_img,
        bpda_img,
        f"{output_dir}/comparison.png"
    )

    # --------------------------------------------------
    # COMPUTE METRICS
    # --------------------------------------------------

    report = []

    report.append(
        f"Original Similarity = {clean_similarity:.6f}\n\n"
    )

    attacks = {
        "FGSM": fgsm_img,
        "I-FGSM": ifgsm_img,
        "BPDA": bpda_img
    }

    for attack_name, adv_img in attacks.items():

        adv_emb = get_embedding(
            model,
            adv_img
        )

        similarity = F.cosine_similarity(
            adv_emb,
            ref_emb
        ).item()

        l2_change = torch.norm(
            adv_emb - clean_emb
        ).item()

        report.append(
            f"{attack_name}\n"
        )

        report.append(
            f"Similarity = {similarity:.6f}\n"
        )

        report.append(
            f"Embedding L2 Change = {l2_change:.6f}\n\n"
        )

        print(
            f"{attack_name} Similarity = {similarity:.4f}"
        )

    report_file = f"{output_dir}/attack_report.txt"

    with open(report_file, "w") as f:
        f.writelines(report)

    print("\nSaved results to:")
    print(output_dir)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    models = build_models("weights")

    model = models["FaceNet"]

    evaluate_image(
        model=model,
        image_path="test.jpg",
        reference_path="reference.jpg",
        output_dir="attack_results"
    )