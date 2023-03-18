## FAQD
# Feature Affinity Assisted Quantized Distillation
Code for Feature Affinity Assisted Knowledge Distillation and Quantization of Deep Neural Networks on Label-Free Data
https://arxiv.org/pdf/2302.10899.pdf

# Example
End-to-end label-free weight quantization
```shell
python quant_distill_cifar100_weight.py --num_bit 4 --label_coef 0 --fine_tuning False
```
Fine-tuning 1W4A full quantization 
```shell
python quant_distill_cifar10_full.py --w_bit 1 --a_bit 4 --fine_tuning True
```
