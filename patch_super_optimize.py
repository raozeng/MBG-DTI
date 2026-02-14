
import os

print("Applying SUPER optimizations for >90% Accuracy...")

# --- 1. Patch architectures.py (Better Interaction + Dropout) ---
arch_path = 'architectures.py'
with open(arch_path, 'r', encoding='utf-8') as f:
    arch_code = f.read()

# 1.1 Increase Dropout to 0.4
if "nn.Dropout(0.2)" in arch_code:
    arch_code = arch_code.replace("nn.Dropout(0.2)", "nn.Dropout(0.4)")
    print(" - Increased Dropout to 0.4")

# 1.2 Improve Interaction Head (Bilinear product)
old_catted = """combined = torch.cat([d_global, p_global], dim=-1)
        score = self.classifier(combined)"""
        
new_catted = """# Advanced Interaction: Concat + Element-wise Product
        interaction = d_global * p_global
        combined = torch.cat([d_global, p_global, interaction], dim=-1)
        score = self.classifier(combined)"""

if "combined = torch.cat([d_global, p_global], dim=-1)" in arch_code:
    arch_code = arch_code.replace(old_catted, new_catted)
    print(" - Enabled Advanced Interaction (Product)")

# 1.3 Update Classifier Input Dimension (hidden*2 -> hidden*3)
# Need to find where self.classifier is defined and change input dim
old_classifier = """        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),"""
new_classifier = """        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),"""

if old_classifier in arch_code:
    arch_code = arch_code.replace(old_classifier, new_classifier)
    print(" - Updated Classifier Input Dimension (x2 -> x3)")

# 1.4 Ensure Sigmoid is REMOVED (Double Check)
old_sigmoid = """            nn.Linear(128, 1),
            nn.Sigmoid()
        )"""
new_sigmoid_block = """            nn.Linear(128, 1)
        )"""

if old_sigmoid in arch_code:
    arch_code = arch_code.replace(old_sigmoid, new_sigmoid_block)
    print(" - Removed Sigmoid Layer (if present)")

with open(arch_path, 'w', encoding='utf-8') as f:
    f.write(arch_code)


# --- 2. Patch train.py (Scheduler + Loss) ---
train_path = 'train.py'
with open(train_path, 'r', encoding='utf-8') as f:
    train_code = f.read()

# 2.1 Add Scheduler Import
if "from torch.optim.lr_scheduler import ReduceLROnPlateau" not in train_code:
    train_code = "from torch.optim.lr_scheduler import ReduceLROnPlateau\n" + train_code

# 2.2 Initialize Scheduler
old_optimizer = """optimizer = optim.Adam(model.parameters(), lr=lr)"""
new_optimizer_block = """optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)"""

if old_optimizer in train_code and "scheduler =" not in train_code:
    train_code = train_code.replace(old_optimizer, new_optimizer_block)
    print(" - Added ReduceLROnPlateau Scheduler")

# 2.3 Step Scheduler in Loop
# Find where validation ends (just before logging)
old_scheduler_step = """            print(f"  Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_result['loss']:.4f}, Val Acc={val_result['acc']:.2f}%")"""
new_scheduler_step = """            print(f"  Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_result['loss']:.4f}, Val Acc={val_result['acc']:.2f}%")
            
            # Step Scheduler
            scheduler.step(val_result['acc'])"""

if old_scheduler_step in train_code and "scheduler.step" not in train_code:
    train_code = train_code.replace(old_scheduler_step, new_scheduler_step)
    print(" - Enabled Scheduler Stepping")

# 2.4 Ensure BCEWithLogitsLoss (Double Check)
# Try to find standard criterion definition and replace if not already smart
old_criterion_simple = """        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)"""

if old_criterion_simple in train_code:
    # We replace it with the smart one
    new_smart_criterion = """        # Auto-weighting (Softened)
        all_labels = dataset.labels
        num_pos = sum(all_labels)
        num_neg = len(all_labels) - num_pos
        # Soften the weight: sqrt or divide by 2 to prevent over-swinging
        pos_weight_val = (num_neg / num_pos) * 0.5 if num_pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val]).to(device)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val:.2f} (Original Ratio: {num_neg/num_pos:.2f})")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-5) # Lower LR for stability
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)"""
    train_code = train_code.replace(old_criterion_simple, new_smart_criterion)
    print(" - Switched to BCEWithLogitsLoss + Auto Weighting")

# 2.5 Ensure Validation uses Sigmoid
old_val_simple = """            predicted = (outputs > 0.5).float()"""
new_val_smart = """            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()"""

if old_val_simple in train_code:
    train_code = train_code.replace(old_val_simple, new_val_smart)
    print(" - Fixed Validation Sigmoid")

with open(train_path, 'w', encoding='utf-8') as f:
    f.write(train_code)

print("SUPER Patch Applied Successfully!")
