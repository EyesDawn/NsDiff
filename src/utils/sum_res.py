import glob, json, os, ast
results = {}
for path in glob.glob('results/runs/NsDiff4/**/output.log', recursive=True):
    parts = path.split('/')
    if len(parts) > 3:
        dataset = parts[3]
        if dataset not in results:
            results[dataset] = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'test_results:' in line:
                    res_str = line.split('test_results:')[1].strip()
                    try:
                        res_dict = ast.literal_eval(res_str)
                        res_dict['path'] = path
                        results[dataset].append(res_dict)
                    except Exception as e:
                        print('Error parsing ' + path + ': ' + str(e))
                    break
with open('results/runs/NsDiff4/NsDiff_resutls.json', 'w') as f:
    json.dump(results, f, indent=4)
print('Done!')
