import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter, calculate_accuracy, calculate_accuracy_pytorch


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, result_path, class_names, no_average,
              output_topk, device, multilabel=False, n_classes=10):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    results = {'results': defaultdict(list)}

    end_time = time.time() 
    

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            #print(targets)
            #print(len(targets))
            
            if multilabel:
                label_targets = torch.Tensor([t[0].numpy() for t in targets])
            else:
                print("targets", targets)
                label_targets = torch.Tensor([t[0] for t in targets])
                #label_targets = torch.Tensor(targets)
            #print(label_targets)
            #other_targets = [[t[0], t[1]] for t in targets]
            #print(other_targets)

            #video_ids, segments = zip(*other_targets)
            targets = label_targets.to(device, non_blocking=True)
            # print("INPUTS", inputs)
            # print("INPUTS size", len(inputs))
            outputs_unnorm = model(inputs)
            outputs = F.softmax(outputs_unnorm, dim=1).cpu()
            acc = calculate_accuracy_pytorch(outputs_unnorm, targets, multilabel=multilabel, n_classes=n_classes)
            #print(len(inputs))
            accuracies.update(acc, inputs.size(0))

            #for j in range(outputs.size(0)):
            #    results['results'][video_ids[j]].append({
            #        'segment': segments[j],
            #        'output': outputs[j]
            #    })
                #print('segment', segments[j], 'output', outputs[j])

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      acc=accuracies))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            #inference_results['results'][video_id] = get_video_results(
             #   average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                #inference_results['results'][video_id].append({
                #    'segment': segment,
                #    'result': result
                #})

    print("accuracies", accuracies.avg)
    inference_results['accuracy'] = accuracies.avg.item()
    
    print("result_path", result_path)
    #print("inference_results", inference_results)

    with open(result_path, 'w') as f:
        json.dump(inference_results, f)
    
    #with open(result_path, 'w') as f:
    #    json.dump({'accuracy': accuracies.avg}, f) #.item()}, f)

    return inference_results
