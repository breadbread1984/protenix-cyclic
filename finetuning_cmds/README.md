第 1 轮：Finetune 主干 + RPE
yaml
crop_size: 648
alpha_pae: 0.0        # 关闭！
alpha_diffusion: 4.0
alpha_distogram: 0.03
freeze_confidence: true  # 冻结置信度头
第 2 轮：Finetune 长序列
yaml
crop_size: 768
alpha_pae: 0.0        # 继续关闭！
freeze_confidence: true
第 3 轮：训练 PAE / 置信度（最后一步）
yaml
crop_size: 768
alpha_pae: 1.0       # 现在才打开 ✅
freeze_trunk: true   # 冻结主干！只训练 PAE 头

but current configuration can only support crop_size: 384, no need for phase 2. just phase 1 and 3

NOTE: must update **load_checkpoint_path** in the config file
