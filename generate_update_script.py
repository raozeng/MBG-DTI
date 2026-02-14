import os

# Files to sync
files_to_sync = [
    'architectures.py',
    'train.py',
    'run.py',
    'dataset.py',
    'run_persistent.sh',
    'collect_results.py'
]

output_script = 'manual_update_v3.py'

content_accumulator = []
content_accumulator.append("import os")
content_accumulator.append("print('Starting Manual Code Update (v3)...')")

for filename in files_to_sync:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        # Escape backslashes and triple quotes for python string literal
        # actually, using raw string r'''...''' is best, but we need to handle if the file contains ''' itself.
        # Simple hack: replace ''' with ' + "''" + ' (concatenate)
        
        safe_content = file_content.replace("'''", "' + \"''\" + '")
        
        content_accumulator.append(f"\n# --- Updating {filename} ---")
        content_accumulator.append(f"print('Updating {filename}...')")
        content_accumulator.append(f"content = r'''{safe_content}'''")
        content_accumulator.append(f"with open('{filename}', 'w', encoding='utf-8') as f:")
        content_accumulator.append(f"    f.write(content)")

content_accumulator.append("\nprint('All files updated successfully!')")

with open(output_script, 'w', encoding='utf-8') as f:
    f.write('\n'.join(content_accumulator))

print(f"Generated {output_script}")
