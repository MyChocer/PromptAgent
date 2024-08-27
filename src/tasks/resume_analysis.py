# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import os
import json
import difflib
import uuid
from enum import Enum
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# TODO: should be import from our project
class ResumeAnalysisDimension(str, Enum):
    UseActionVerb = "Use Action Verbs"
    MethodologyExplanation = "Methodology Explanation"
    EmphasizeAccomplishment = "Emphasize Accomplishment"
    QuantificationOfAchievements = "Quantification of Achievements"
    UseDiverseActionVerbs = "Use Diverse Action Verbs"
    SpellingAndVerbTenses = "Spelling & Verb Tenses"
    AppropriateBulletLength = "Appropriate Bullet Length"
    AvoidanceOfBuzzwordsAndCliches = "Avoidance of Buzzwords and Cliches"
    AvoidPersonalPronouns = "Avoid Personal Pronouns"
    SectionCompleteness = "Section Completeness"
    AppropriateContentLength = "Appropriate Content Length"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, ResumeAnalysisDimension))
    
    @staticmethod
    def from_str(value: str):
        for item in ResumeAnalysisDimension:
            if item.value == value:
                return item
        raise f"Invalid DimensionType for {value}"

@dataclass_json
@dataclass
class ResumeAnalysisResult:
    items: dict[str, list[str]]

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None, 
                 
                 task_name = 'resume_analysis',
                 task_description = 'resume_analysis',
                 data_dir=None, 
                 seed=None,  
                 post_instruction=False, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):

        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        TaskDataset=TaskDataset,
                        option_num=option_num,
                        )
                
    def load_task_dataset(self, data_dir):
        dataset = self._load_json_file(data_dir)
        
        example_folder = dataset['folder']
        example_paths = dataset['examples']

        examples = []
        for path in example_paths:
            with open(os.path.join(example_folder, path), 'r') as file:
                resume = json.load(file)

            formatted_example = {
                'question': self._format_question(resume),
                'answer': json.dumps(self._format_answer(resume).to_dict()) # this is a dataloader restrcition that not allowing to return a customzie class
            }
            examples.append(formatted_example)

        return examples
    
    def _format_question(self, resume: dict) -> str:
        text = ""
        experience_id = 0
        pre_experience = None
        for report in resume['details']:
            if report['originalContent'] is None or report['originalContent'] == "":
                continue
            # @TODO: need to check the path is correct
            if "path" not in report:
                cur_experience = str(uuid.uuid4())
            else:
                cur_experience = '.'.join(report['path'].split(".")[:-1])

            if cur_experience != pre_experience:
                text += "\n"
                text += f"## Experience {experience_id}\n"

                experience_id += 1
                pre_experience = cur_experience
            
            text += f"- {report['originalContent']}\n"
        return text

    def _format_answer(self, resume: dict) -> ResumeAnalysisResult:
        _all = {}

        for report in resume['details']:
            bullet_point = report['originalContent']
            _all[bullet_point] = [issue['dimensionType'] for issue in report['issues']]

        return ResumeAnalysisResult(items=_all)

    def build_forward_prompts_completion(self, questions: list[str], cur_propmt: str) -> list[str]:
        '''
        Optional: <task specific>
        The format of combining question and prompts.
        '''
        if self.post_instruction:
            raise NotImplementedError("This task does not support post_instruction")
        
        prompts = []
        for question in questions:
            prompts.append(f'{cur_propmt}\n Here is the resume: {question}\n')
        
        return prompts

    def clean_labels(self, labels: list[ResumeAnalysisResult]) -> list[ResumeAnalysisResult]:
        return labels
    
    def clean_response(self, response: str, n_columns: int = 4) -> ResumeAnalysisResult:
        result = {}

        for line in response.split("\n"):
            split_line = self._seperate_table_line(line, n_columns)
            if split_line:
                origin_bullet_point_content, _, violate_dimensions, _ = split_line  
                dimension_unit = self._string_to_dimension(violate_dimensions)
                
                # This is used to remove the header case
                if len(dimension_unit) > 0:
                    result[origin_bullet_point_content] = dimension_unit
        
        return ResumeAnalysisResult(items=result)
    
    def cal_correct(self, preds: list[ResumeAnalysisResult], labels: list[str], thr: float = 0.8): # this is a dataloader restrcition that not allowing to return a customzie class
        comparisons = []
        for p, l in zip(preds, labels):
            l = ResumeAnalysisResult.from_json(l)

            recall, precision = self._evaluate(p, l)
            print("==================================")
            print(f"Recall: {recall}, Precision: {precision}")
            print("==================================")
            metric = (recall + precision) / 2
            if metric >= thr:
                comparisons.append(1)
            else:
                comparisons.append(0)
        return comparisons
    
    def _evaluate(self, pred: ResumeAnalysisResult, label: ResumeAnalysisResult, thr: float = 0.9) -> (float, float):
        recall = self._cal_recall(pred, label)
        precision = self._cal_precision(pred, label)
        return (recall, precision)
    
    def _cal_recall(self, pred: ResumeAnalysisResult, label: ResumeAnalysisResult) -> float:
        correct_case = 0
        all_case = 0

        for label_bullet_point, label_dimensions in label.items.items():
            all_case += len(label_dimensions)
            for pred_bullet_point, pred_dimensions in pred.items.items():
                if is_same_text(label_bullet_point, pred_bullet_point):
                    correct_case += len([item for item in label_dimensions if item in pred_dimensions])

        if all_case == 0:
            return 1.0
        return correct_case / all_case
    
    def _cal_precision(self, pred: ResumeAnalysisResult, label: ResumeAnalysisResult) -> float:
        correct_case = 0
        all_case = 0

        for pred_bullet_point, pred_dimensions in pred.items.items():
            all_case += len(pred_dimensions)
            for label_bullet_point, label_dimensions in label.items.items():
                if is_same_text(label_bullet_point, pred_bullet_point):
                    correct_case += len([item for item in pred_dimensions if item in label_dimensions])

        if all_case == 0:
            return 1.0
        return correct_case / all_case
        
    def _seperate_table_line(self, line: str, n_columns: int, separator: str="|") -> list[str]:
        if not line.strip().startswith(separator):
            return None
        
        #  To avoid seperator in real content, add some space to the separaotr
        separator = " " + separator + " "
        line = " " + line + " "

        split_line = line.split(separator)

        if len(split_line) != n_columns + 2:
            return None
        
        return [item.strip() for item in split_line[1:-1]] 

    def _string_to_dimension(self, raw_string: str, delimiter: str = ",") -> list[str]:
        dimensions = raw_string.split(delimiter)

        output = []
        for dimension in dimensions:
            for item in ResumeAnalysisDimension:
                if is_same_text(item.value, dimension.strip()):
                    output.append(item.value)
                    break
        return output
    

def is_same_text(s1, s2, thr = 0.9):
    if s1.lower() == s2.lower():
        return True
    
    len_s1, len_s2 = len(s1), len(s2)

    if max(len_s1, len_s2) == 0:
        return True
            
    if float(min(len_s1, len_s2)) / max(len_s1, len_s2) < thr:
        return False
    
    string_comparor = difflib.Differ()

    all_char_result = list(string_comparor.compare(s1.lower(), s2.lower()))
    match_char_result = [item for item in all_char_result if item.startswith(" ")]

    return len(match_char_result) / len(all_char_result) >= thr
