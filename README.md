## FAQD
# Feature Affinity Assisted Quantized Distillation

Code for 

Feature Affinity Assisted Knowledge Distillation and Quantization of Deep Neural Networks on Label-Free Data

https://arxiv.org/pdf/2302.10899.pdf

# Model Definitions:
Regular ResNet: weight_quantization/resnet_cifar.py

Quantized ReLU with straight-through estimator: full_quantization/quan_uni_type.py

ResNet with quantized activation: full_quantization/resnet_type_cifar.py


# Example
End-to-end label-free 4-bit weight quantization on CIFAR-100 dataset
```shell
python quant_distill_cifar100_weight.py --num_bit 4 --label_coef 0 --fine_tuning False
```
Fine-tuning 1W4A full quantization on CIFAR-10 dataset
```shell
python quant_distill_cifar10_full.py --w_bit 1 --a_bit 4 --fine_tuning True
```
End-to-end 4-bit weight quantization on CIFAR-10 dataset with fast feature affinity loss (k=15)
```shell
python quant_distill_cifar10_full.py --num_bit 4 --fa_type FFA --num_ensemble 15
```
