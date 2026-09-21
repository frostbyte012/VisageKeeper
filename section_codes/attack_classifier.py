import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms

from adversarial_face_eval import *

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# SAVE IMAGE
# ============================================================

def save_tensor_image(x, path):

    img = denorm(x)

    img = img.squeeze(0)

    img = img.clamp(0,1)

    img = transforms.ToPILImage()(img.cpu())

    img.save(path)


# ============================================================
# SAVE NOISE
# ============================================================

def save_noise(clean, adv, path):

    diff = (adv-clean).abs()

    diff = diff.squeeze(0)

    diff = diff.cpu().numpy()

    diff = np.transpose(diff,(1,2,0))

    diff = diff/(diff.max()+1e-8)

    plt.imshow(diff)

    plt.axis("off")

    plt.savefig(path,bbox_inches="tight")

    plt.close()




def get_logits(model, x):

    backbone = model.backbone

    x = to_vgg_space(x)

    x = backbone.maxpool(
        backbone.relu(
            backbone.bn1(
                backbone.conv1(x)
            )
        )
    )

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)

    x = backbone.avgpool(x)

    feat = x.flatten(1)

    logits = backbone.fc(feat)

    return logits


# ============================================================
# PREDICTION
# ============================================================

def predict(model,x):

    logits = get_logits(model,x)

    probs = F.softmax(
        logits,
        dim=1
    )

    conf,pred = probs.max(1)

    return (
        pred.item(),
        conf.item()
    )


# ============================================================
# FGSM CLASSIFICATION
# ============================================================

def fgsm_classification_attack(
        model,
        x,
        epsilon=0.10):

    pred,_ = predict(model,x)

    y = torch.tensor(
        [pred],
        device=x.device
    )

    x_adv = x.clone()

    x_adv.requires_grad_(True)

    logits = get_logits(
        model,
        x_adv
    )

    loss = F.cross_entropy(
        logits,
        y
    )

    loss.backward()

    eps_n = _en(
        epsilon,
        x.device
    )

    with torch.no_grad():

        x_adv = x + eps_n * x_adv.grad.sign()

        x_adv = _project(
            x_adv,
            x,
            eps_n
        )

    return x_adv


# ============================================================
# IFGSM
# ============================================================

def _en(epsilon, device):
    return epsilon / torch.tensor(
        IMAGENET_STD,
        dtype=torch.float32,
        device=device
    ).view(1,3,1,1)


def _project(x_adv, x_orig, eps_n):

    delta = (x_adv - x_orig).clamp(
        -eps_n,
        eps_n
    )

    return renorm(
        denorm(
            x_orig + delta
        ).clamp(0,1)
    ).detach()

def ifgsm_classification_attack(
        model,
        x,
        epsilon=0.10,
        steps=10):

    pred,_ = predict(model,x)

    y = torch.tensor(
        [pred],
        device=x.device
    )

    eps_n = _en(
        epsilon,
        x.device
    )

    alpha_n = _en(
        epsilon/steps,
        x.device
    )

    x_adv = x.clone()

    for _ in range(steps):

        x_adv.requires_grad_(True)

        logits = get_logits(
            model,
            x_adv
        )

        loss = F.cross_entropy(
            logits,
            y
        )

        loss.backward()

        with torch.no_grad():

            x_adv = x_adv + alpha_n * x_adv.grad.sign()

            x_adv = _project(
                x_adv,
                x,
                eps_n
            )

    return x_adv


# ============================================================
# BPDA
# ============================================================

def bpda_classification_attack(
        model,
        x,
        epsilon=0.10,
        steps=10):

    pred,_ = predict(model,x)

    y = torch.tensor(
        [pred],
        device=x.device
    )

    eps_n = _en(
        epsilon,
        x.device
    )

    alpha_n = _en(
        epsilon/steps,
        x.device
    )

    x_adv = x.clone()

    for _ in range(steps):

        x_adv.requires_grad_(True)

        with torch.no_grad():

            blur = F.avg_pool2d(
                x_adv.detach(),
                3,
                1,
                1
            )

        x_bpda = x_adv + (
            blur - x_adv
        ).detach()

        logits = get_logits(
            model,
            x_bpda
        )

        loss = F.cross_entropy(
            logits,
            y
        )

        loss.backward()

        with torch.no_grad():

            x_adv = x_adv + alpha_n * x_adv.grad.sign()

            x_adv = _project(
                x_adv,
                x,
                eps_n
            )

    return x_adv


# ============================================================
# MAIN
# ============================================================

models = build_models("weights")

model = models["ResNet50-FT"]

image_path = "test.jpg"

output_dir = "attack_results"

os.makedirs(
    output_dir,
    exist_ok=True
)

x = load_image(
    image_path
)

clean_pred,clean_conf = predict(
    model,
    x
)

fgsm = fgsm_classification_attack(
    model,
    x
)

ifgsm = ifgsm_classification_attack(
    model,
    x
)

bpda = bpda_classification_attack(
    model,
    x
)

results = {
    "Clean":x,
    "FGSM":fgsm,
    "IFGSM":ifgsm,
    "BPDA":bpda
}

report = []

for name,img in results.items():

    pred,conf = predict(
        model,
        img
    )

    save_tensor_image(
        img,
        f"{output_dir}/{name}.png"
    )

    report.append(
        f"{name}\n"
    )

    report.append(
        f"Class ID : {pred}\n"
    )

    report.append(
        f"Confidence : {conf:.6f}\n\n"
    )

save_noise(
    x,
    fgsm,
    f"{output_dir}/FGSM_noise.png"
)

save_noise(
    x,
    ifgsm,
    f"{output_dir}/IFGSM_noise.png"
)

save_noise(
    x,
    bpda,
    f"{output_dir}/BPDA_noise.png"
)

with open(
    f"{output_dir}/classification_results.txt",
    "w"
) as f:

    f.writelines(report)

print(
    "\nResults saved in attack_results/"
)