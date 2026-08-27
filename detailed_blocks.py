import re

files = [
    "explaining_dpo_loss_with_qwen_example.md",
    "explaining_grpo_loss_with_qwen2.5.md",
    "explaining_kto_loss_with_qwen2.md",
    "explaining_reward_model_loss_with_qwen3.md"
]

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.splitlines()
    sec_idx = -1
    for idx, line in enumerate(lines):
        if "Simple PyTorch Implementation" in line:
            sec_idx = idx
            break
            
    print(f"\n==================== {fpath} ====================")
    if sec_idx != -1:
        in_code = False
        lang = ""
        code_blocks = []
        current_block = []
        for i in range(sec_idx, len(lines)):
            line = lines[i]
            if line.startswith("```"):
                if in_code:
                    # Closing
                    code_blocks.append((lang, "\n".join(current_block)))
                    current_block = []
                    in_code = False
                    lang = ""
                else:
                    # Opening
                    in_code = True
                    lang = line[3:].strip()
            elif in_code:
                current_block.append(line)
        
        for k, (l, code) in enumerate(code_blocks):
            print(f"Block {k} ({l}): line count: {len(code.splitlines())}")
            # print first 3 and last 3 lines
            clines = code.splitlines()
            if len(clines) > 6:
                print("  " + "\n  ".join(clines[:3]))
                print("  ...")
                print("  " + "\n  ".join(clines[-3:]))
            else:
                print("  " + "\n  ".join(clines))

