# main.py (REBAスコア計算バックエンド)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal
import math
import traceback

# CORSミドルウェアのインポート
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="REBA Evaluation API v2")

origins = ["https://reba-manus.onrender.com"] # 開発用に全て許可。本番環境では適切に設定してください。

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mediapipe Pose Landmark Indices ---
# (https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-card)
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# --- モデル定義 ---
class Landmark(BaseModel):
    x: float
    y: float
    z: Optional[float] = None
    visibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class ManualInputs(BaseModel):
    neck_rotation: bool = Field(default=False)
    neck_lateral_bending: bool = Field(default=False)
    trunk_rotation: bool = Field(default=False)
    trunk_lateral_bending: bool = Field(default=False)
    load_force: Literal[0, 1, 2] = Field(default=0)
    shock_force: bool = Field(default=False)
    upper_arm_abduction: bool = Field(default=False)
    shoulder_raised: bool = Field(default=False)
    forearm_rotation: bool = Field(default=False) # 前腕の回旋 (Supination/Pronation)
    wrist_ulnar_radial_deviation: bool = Field(default=False) # 手首の尺屈/橈屈
    coupling_score: Literal[0, 1, 2, 3] = Field(default=0)
    activity_score: Literal[0, 1, 2] = Field(default=0) # ユーザーは0,1,2で指定

class RebaInput(BaseModel):
    landmarks: List[Landmark]
    manual_inputs: ManualInputs
    filming_side: Literal["left", "right"] = Field(default="left")

class RebaScoreResponse(BaseModel):
    reba_score: Optional[float] = None
    risk_level: str = "評価中"
    # 詳細スコア（デバッグや詳細表示用）
    neck_score: Optional[int] = None
    trunk_score: Optional[int] = None
    legs_score: Optional[int] = None
    table_a_score: Optional[int] = None
    upper_arm_score: Optional[int] = None
    lower_arm_score: Optional[int] = None
    wrist_score: Optional[int] = None
    table_b_score: Optional[int] = None
    table_c_score: Optional[int] = None
    is_high_risk: bool = False
    error_message: Optional[str] = None
    angles: Optional[Dict[str, Optional[float]]] = None # 計算された角度

# --- Helper Functions for Vector and Angle Calculations ---
MIN_VISIBILITY_THRESHOLD = 0.3 # visibilityの閾値

def get_landmark(landmarks: List[Landmark], index: int) -> Optional[Landmark]:
    if 0 <= index < len(landmarks):
        lm = landmarks[index]
        # visibilityが低い場合はNoneを返す（角度計算で考慮される）
        if lm.visibility is not None and lm.visibility < MIN_VISIBILITY_THRESHOLD:
            return None
        return lm
    return None

def calculate_vector_2d(p1: Landmark, p2: Landmark) -> Optional[Dict[str, float]]:
    if p1 and p2:
        return {"x": p2.x - p1.x, "y": p2.y - p1.y}
    return None

def vector_magnitude_2d(v: Dict[str, float]) -> float:
    return math.sqrt(v["x"]**2 + v["y"]**2)

def dot_product_2d(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    return v1["x"] * v2["x"] + v1["y"] * v2["y"]

def angle_between_vectors_2d(v1_p1: Landmark, v1_p2: Landmark, v2_p1: Landmark, v2_p2: Landmark) -> Optional[float]:
    # 3点から2つのベクトルを形成し、その間の角度を計算
    # v1 = p1 -> p2, v2 = p1 -> p3 のような場合、p1が共通点
    # ここでは v1 = v1_p2 - v1_p1, v2 = v2_p2 - v2_p1 として、v1とv2の間の角度を求める
    # 実際には3点 P_center, P_a, P_b があり、 P_center-P_a と P_center-P_b の間の角度を求めることが多い
    # この関数はより汎用的に2つのベクトル間の角度を計算する
    # 関節角度計算では、calculate_angle_3points_2d を使う方が直感的
    vec1 = calculate_vector_2d(v1_p1, v1_p2)
    vec2 = calculate_vector_2d(v2_p1, v2_p2)

    if not vec1 or not vec2:
        return None

    mag1 = vector_magnitude_2d(vec1)
    mag2 = vector_magnitude_2d(vec2)

    if mag1 * mag2 < 1e-6: # ゼロ除算や非常に小さいマグニチュードを避ける
        return 0.0 # または None, 状況による

    dot = dot_product_2d(vec1, vec2)
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta)) # 数値誤差による範囲外の値を補正
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def calculate_angle_3points_2d(p_center: Landmark, p_a: Landmark, p_b: Landmark) -> Optional[float]:
    """ 3つのランドマークから関節角度を計算 (p_centerが関節点) """
    if not p_center or not p_a or not p_b:
        return None

    # ベクトル v_center_a と v_center_b を作成
    v_ca = calculate_vector_2d(p_center, p_a)
    v_cb = calculate_vector_2d(p_center, p_b)

    if not v_ca or not v_cb:
        return None

    mag_ca = vector_magnitude_2d(v_ca)
    mag_cb = vector_magnitude_2d(v_cb)

    if mag_ca * mag_cb < 1e-6:
        return 0.0

    dot = dot_product_2d(v_ca, v_cb)
    cos_theta = dot / (mag_ca * mag_cb)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def calculate_angle_with_vertical(p1: Landmark, p2: Landmark, invert_y: bool = False) -> Optional[float]:
    """ p1からp2へのベクトルと垂直線との角度を計算 """
    if not p1 or not p2:
        return None

    vec = calculate_vector_2d(p1, p2)
    if not vec: return None

    # 垂直ベクトル (Y軸方向、MediapipeのYは下向きが正)
    # 画面上向きを0度とする場合、(0, -1) を使う
    # Y軸が下向きなので、(0,1)が下向きの垂直線。角度の解釈に注意。
    # ここでは、Y軸の正方向（下向き）を基準とする。
    # 体幹の前屈/後屈などでは、Y軸との角度が重要。
    # invert_y = True の場合、Y軸を反転して計算（例：p2.y - p1.y の代わりに p1.y - p2.y）
    # 通常、Y座標は下が正なので、(p2.y - p1.y) が正なら下向きのベクトル成分
    
    # 基準となる垂直ベクトル（Y軸の負の方向、つまり上向き）
    vertical_vec = {"x": 0, "y": -1}
    if invert_y: # 通常のMediapipe座標系では不要なことが多い
        vec["y"] *= -1

    mag_vec = vector_magnitude_2d(vec)
    mag_vertical = 1.0 # vector_magnitude_2d(vertical_vec)

    if mag_vec * mag_vertical < 1e-6:
        return 0.0

    dot = dot_product_2d(vec, vertical_vec)
    cos_theta = dot / (mag_vec * mag_vertical)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    # X成分の符号で左右どちらに傾いているか判定し、0-360度の範囲にすることも可能だが、
    # REBAでは通常0-180度の屈曲/伸展角度が使われる。
    # 例えば、体幹の前屈は正、後屈は負（または基準からの変化）として扱うことが多い。
    # ここでは単純な角度を返す。
    return angle_deg

# --- REBA Score Calculation Logic ---
# (REBAスコアシートに基づくテーブルとロジック)

# 表A: 体幹、頸部、脚のスコア
# table_a_trunk[trunk_pos_score][neck_pos_score][leg_pos_score]
# これは非常に巨大なテーブルになるため、実際にはルールベースで計算するか、
# 既存のREBA計算ライブラリのロジックを参照する。
# ここでは簡略化のため、各部位のスコアを個別に計算し、それらを組み合わせる方式を想定。

# REBAのスコアテーブルは複雑なので、ここでは主要な関節角度から各部位のスコアを算出する関数群を定義

def get_neck_score(neck_angle_f_e: Optional[float], neck_lat_bend: bool, neck_rot: bool) -> int:
    score = 0
    if neck_angle_f_e is None: return 1 # 不明な場合は中間的なスコア

    # 頸部屈曲/伸展
    if neck_angle_f_e < 0: # 伸展 (後屈)
        score = 2
    elif 0 <= neck_angle_f_e <= 20:
        score = 1
    else: # neck_angle_f_e > 20 (屈曲)
        score = 2
        if neck_angle_f_e > 45: # 過度な屈曲のイメージ（REBAシートの図を参照）
             score = 3 # 仮（REBAの正確な閾値を確認する必要あり）

    if neck_lat_bend or neck_rot:
        score += 1
    return score

def get_trunk_score(trunk_angle_f_e: Optional[float], trunk_lat_bend: bool, trunk_rot: bool) -> int:
    score = 0
    if trunk_angle_f_e is None: return 2 # 不明な場合は中間的なスコア

    # 体幹屈曲/伸展
    # 0度が直立、正が前屈、負が後屈と仮定
    abs_angle = abs(trunk_angle_f_e)
    if abs_angle <= 5: # 直立に近い
        score = 1
    elif abs_angle <= 20:
        score = 2
    elif abs_angle <= 60:
        score = 3
    else: # abs_angle > 60
        score = 4
    
    if trunk_angle_f_e < -5 : # 明確な後屈の場合 (REBAシートの図と閾値を確認)
        score = 2 # 後屈のスコア（仮）
        if trunk_angle_f_e < -20:
            score = 3 # 過度な後屈（仮）

    if trunk_lat_bend or trunk_rot:
        score += 1
    return score

def get_legs_score(knee_angle: Optional[float], hip_angle: Optional[float], bilateral_support: bool = True, weight_on_legs_unilateral_or_unstable: bool = False) -> int:
    # REBAの脚スコアは複雑。座位、立位、片足、膝の屈曲度合いなど。
    # ここでは簡略化。bilateral_supportは両足支持を意味する。
    score = 1 # 基本: 両足で体重を支え、歩行または座位
    if knee_angle is None: return score # 不明な場合は基本スコア

    if not bilateral_support or weight_on_legs_unilateral_or_unstable: # 片足支持や不安定な場合
        score = 2

    # 膝の屈曲による追加スコア
    if 30 <= knee_angle <= 60:
        score += 1
    elif knee_angle > 60:
        score += 2
    return score

def get_upper_arm_score(upper_arm_angle_f_e: Optional[float], shoulder_raised: bool, upper_arm_abducted: bool, gravity_assisted: bool = False) -> int:
    score = 0
    if upper_arm_angle_f_e is None: return 2 # 不明な場合は中間的なスコア

    # 上腕の屈曲/伸展 (0度が体側、正が前方挙上、負が後方伸展)
    if upper_arm_angle_f_e < -20: # 後方伸展
        score = 2
    elif -20 <= upper_arm_angle_f_e <= 20:
        score = 1
    elif 20 < upper_arm_angle_f_e <= 45:
        score = 2
    elif 45 < upper_arm_angle_f_e <= 90:
        score = 3
    else: # upper_arm_angle_f_e > 90
        score = 4

    if shoulder_raised: score += 1
    if upper_arm_abducted: score += 1
    if gravity_assisted: score -=1 # 重力による補助がある場合（例：腕をテーブルに置いている）
    return max(1, score) # 最低スコアは1

def get_lower_arm_score(lower_arm_angle_flex: Optional[float], forearm_rotated: bool) -> int:
    # 肘の屈曲角度 (0度が完全伸展、180度が完全屈曲)
    score = 0
    if lower_arm_angle_flex is None: return 1 # 不明な場合は中間的なスコア

    if 60 <= lower_arm_angle_flex <= 100:
        score = 1
    else: # < 60 or > 100
        score = 2
    
    # REBAでは前腕の回旋は手入力で補うことが多いが、ここでは引数で受ける
    # if forearm_rotated: score +=1 # REBAのスコアシートでは前腕のスコアに回旋は直接影響しないことが多い。手首で考慮。
    return score

def get_wrist_score(wrist_angle_f_e: Optional[float], wrist_dev: bool, wrist_twist: bool) -> int:
    # 手首の屈曲/伸展 (0度が中間位)
    score = 0
    if wrist_angle_f_e is None: return 1 # 不明な場合は中間的なスコア

    if abs(wrist_angle_f_e) <= 15:
        score = 1
    else: # > 15度
        score = 2

    if wrist_dev or wrist_twist: # 尺屈/橈屈 または ねじり
        score += 1
    return score

# REBA Table A, B, C (非常に複雑なため、実際のプロジェクトでは詳細な参照が必要)
# ここでは仮のスコアテーブルや計算ロジックを使用
TABLE_A_SCORES = [
    # このテーブルは非常に巨大で、[Trunk][Neck][Legs] の組み合わせで決まる
    # 例: table_a_scores[trunk_score-1][neck_score-1][legs_score-1]
    # 実際のREBA評価シートを参照して正確に実装する必要がある
    # 以下はダミーの構造
    [[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]], # Trunk 1
    [[2,3,4,5], [3,4,5,6], [4,5,6,7], [5,6,7,8]], # Trunk 2
    [[3,4,5,6], [4,5,6,7], [5,6,7,8], [6,7,8,9]], # Trunk 3
    [[4,5,6,7], [5,6,7,8], [6,7,8,9], [7,8,9,10]],# Trunk 4
    [[5,6,7,8], [6,7,8,9], [7,8,9,10], [8,9,10,11]] # Trunk 5 (max trunk score 5と仮定)
]

TABLE_B_SCORES = [
    # [UpperArm][LowerArm][Wrist]
    # 以下はダミーの構造
    [[1,2,2,3], [2,3,3,4], [3,4,4,5]], # UpperArm 1
    [[2,3,3,4], [3,4,4,5], [4,5,5,6]], # UpperArm 2
    [[3,3,4,5], [4,4,5,6], [5,5,6,7]], # UpperArm 3
    [[4,4,5,6], [5,5,6,7], [6,6,7,8]], # UpperArm 4
    [[5,5,6,7], [6,6,7,8], [7,7,8,9]], # UpperArm 5
    [[6,6,7,8], [7,7,8,9], [8,8,9,9]]  # UpperArm 6 (max upper arm score 6と仮定)
]

TABLE_C_SCORES = [
    # [TableAScoreWithLoad][TableBScoreWithCoupling]
    # 以下はダミーの構造 (12x12のテーブルになることが多い)
    # サイズを小さくするため、TableA, TableBのスコア範囲を仮定
    # Max Table A (with load) = 12, Max Table B (with coupling) = 12
    [1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7 ],
    [1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8 ],
    [2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8 ],
    [3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9 ],
    [4,  4,  4,  5,  6,  7,  8,  8,  9,  9, 10, 10],
    [5,  5,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10],
    [6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11],
    [7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],
    [8,  8,  8,  9,  9, 10, 10, 10, 10, 11, 11, 12],
    [9,  9,  9,  9, 10, 10, 11, 11, 11, 12, 12, 12],
    [10,10,10,10,11, 11, 11, 12, 12, 12, 12, 12],
    [11,11,11,11,12, 12, 12, 12, 12, 12, 12, 12],
    [12,12,12,12,12, 12, 12, 12, 12, 12, 12, 12]
]

def get_table_a_score(trunk_s, neck_s, legs_s):
    # テーブルのインデックスは0からなので、スコアから1を引く
    # スコアがテーブルの範囲を超える場合はクリップする
    ts = max(0, min(trunk_s - 1, len(TABLE_A_SCORES) - 1))
    ns = max(0, min(neck_s - 1, len(TABLE_A_SCORES[0]) - 1))
    ls = max(0, min(legs_s - 1, len(TABLE_A_SCORES[0][0]) - 1))
    try:
        return TABLE_A_SCORES[ts][ns][ls]
    except IndexError:
        # print(f"Table A lookup error: T={trunk_s}({ts}), N={neck_s}({ns}), L={legs_s}({ls})")
        return 8 # エラー時は高いスコアを返すなど、適切なエラー処理

def get_table_b_score(upper_arm_s, lower_arm_s, wrist_s):
    uas = max(0, min(upper_arm_s - 1, len(TABLE_B_SCORES) - 1))
    las = max(0, min(lower_arm_s - 1, len(TABLE_B_SCORES[0]) - 1))
    ws = max(0, min(wrist_s - 1, len(TABLE_B_SCORES[0][0]) - 1))
    try:
        return TABLE_B_SCORES[uas][las][ws]
    except IndexError:
        # print(f"Table B lookup error: UA={upper_arm_s}({uas}), LA={lower_arm_s}({las}), W={wrist_s}({ws})")
        return 8

def get_table_c_score(table_a_final_s, table_b_final_s):
    # Table Cのインデックスも0から。スコアは1から始まる想定。
    # REBAのTable Cは最大12x12のことが多い。
    # ここではTABLE_C_SCORESの定義に合わせてクリップする。
    tas = max(0, min(table_a_final_s - 1, len(TABLE_C_SCORES) - 1))
    tbs = max(0, min(table_b_final_s - 1, len(TABLE_C_SCORES[0]) - 1))
    try:
        return TABLE_C_SCORES[tas][tbs]
    except IndexError:
        # print(f"Table C lookup error: A={table_a_final_s}({tas}), B={table_b_final_s}({tbs})")
        return 12 # エラー時は最大スコア

# --- API Endpoint ---
@app.post("/calculate_reba", response_model=RebaScoreResponse)
async def calculate_reba_endpoint(reba_input: RebaInput):
    landmarks = reba_input.landmarks
    manual = reba_input.manual_inputs
    filming_side = reba_input.filming_side
    calculated_angles = {}

    try:
        # ランドマーク取得 (撮影側に応じて左右反転)
        # 簡単のため、ここでは常に左側を基準とする。右側撮影の場合はX座標を反転するなどの処理が必要だが、
        # Mediapipeのランドマーク自体が体の部位を示しているので、インデックスを使い分ける方が確実。
        # 例: 左肩はLEFT_SHOULDER, 右肩はRIGHT_SHOULDER

        # 基準となるランドマークを選択
        shoulder = get_landmark(landmarks, LEFT_SHOULDER if filming_side == "left" else RIGHT_SHOULDER)
        other_shoulder = get_landmark(landmarks, RIGHT_SHOULDER if filming_side == "left" else LEFT_SHOULDER)
        hip = get_landmark(landmarks, LEFT_HIP if filming_side == "left" else RIGHT_HIP)
        other_hip = get_landmark(landmarks, RIGHT_HIP if filming_side == "left" else LEFT_HIP)
        knee = get_landmark(landmarks, LEFT_KNEE if filming_side == "left" else RIGHT_KNEE)
        ankle = get_landmark(landmarks, LEFT_ANKLE if filming_side == "left" else RIGHT_ANKLE)
        elbow = get_landmark(landmarks, LEFT_ELBOW if filming_side == "left" else RIGHT_ELBOW)
        wrist = get_landmark(landmarks, LEFT_WRIST if filming_side == "left" else RIGHT_WRIST)
        
        # 頸部と体幹の基準点 (両肩の中点、両腰の中点)
        mid_shoulder = None
        if shoulder and other_shoulder:
            mid_shoulder = Landmark(x=(shoulder.x + other_shoulder.x)/2, y=(shoulder.y + other_shoulder.y)/2, visibility=min(shoulder.visibility or 0, other_shoulder.visibility or 0) if shoulder.visibility is not None and other_shoulder.visibility is not None else None)
        
        mid_hip = None
        if hip and other_hip:
            mid_hip = Landmark(x=(hip.x + other_hip.x)/2, y=(hip.y + other_hip.y)/2, visibility=min(hip.visibility or 0, other_hip.visibility or 0) if hip.visibility is not None and other_hip.visibility is not None else None)

        nose = get_landmark(landmarks, NOSE)
        ear = get_landmark(landmarks, LEFT_EAR if filming_side == "left" else RIGHT_EAR)

        # --- 角度計算 ---
        # 頸部屈曲/伸展: (耳-肩)ベクトルと(肩-腰)ベクトルのなす角度、または(鼻-両肩中点)と(両肩中点-垂直線)の角度など
        # REBAの定義では、頸部は体幹に対する相対角度も重要。
        # ここでは簡略化のため、(耳-肩)と(肩-腰)の角度を計算。
        # または、(Nose - MidShoulder) と MidShoulderに対する垂直線の角度。
        # 既存コードでは頸部角度を体幹基準に修正とあったので、そのロジックを再現する必要がある。
        # ひとまず、(Ear-Shoulder)と垂直線の角度で代用。
        neck_angle_f_e = None
        if ear and shoulder: # 頸部の角度は、耳、肩、腰のランドマークから計算することが多い
            # 垂直線との角度で計算 (0度が直立、正が前屈)
            # 肩と耳を結ぶ線が垂直線となす角度
            neck_angle_f_e = calculate_angle_with_vertical(shoulder, ear)
            if neck_angle_f_e is not None and ear.y < shoulder.y : # 耳が肩より上（通常）
                 pass # そのままの角度
            elif neck_angle_f_e is not None: # 耳が肩より下（特殊な姿勢）
                 neck_angle_f_e = 180 - neck_angle_f_e # 角度の補正が必要な場合
            # 頸部の屈曲は20度までがスコア1。それ以上でスコア2。
            # 0-20度が自然な範囲。ここでは単純に垂直からの角度。
            # REBAの図では、頭が前に傾くほどスコアが上がる。
            # calculate_angle_with_vertical は上向きを0度とするので、
            # 頭が前に倒れると角度は大きくなる（0-90度）。後ろに倒れると90度以上。
            # REBAの定義に合わせて調整が必要。
            # 仮に、0度が直立、前屈が正、後屈が負とする。
            # 画面の上方向を基準(0, -1)とした場合、(shoulder-ear)ベクトルとの角度。
            # (ear.y < shoulder.y) なら前屈のイメージ。 (ear.y > shoulder.y) なら後屈。
            if neck_angle_f_e is not None:
                if ear.y < shoulder.y: # 前屈
                    calculated_angles["neck_flexion_extension"] = neck_angle_f_e
                else: # 後屈
                    calculated_angles["neck_flexion_extension"] = -neck_angle_f_e # 符号で表現
        
        # 体幹屈曲/伸展: (肩-腰)ベクトルと垂直線の角度
        trunk_angle_f_e = None
        if mid_shoulder and mid_hip:
            trunk_angle_f_e = calculate_angle_with_vertical(mid_hip, mid_shoulder)
            # 0度が直立。mid_shoulderがmid_hipより前(X座標が小さいなど、撮影方向による)なら前屈。
            # Y座標で判断：mid_shoulder.y < mid_hip.y なら前屈。
            if trunk_angle_f_e is not None:
                if mid_shoulder.y < mid_hip.y: # 前屈
                    calculated_angles["trunk_flexion_extension"] = trunk_angle_f_e
                else: # 後屈
                    calculated_angles["trunk_flexion_extension"] = -trunk_angle_f_e
        elif shoulder and hip: # 片側だけでも計算
             trunk_angle_f_e = calculate_angle_with_vertical(hip, shoulder)
             if trunk_angle_f_e is not None:
                if shoulder.y < hip.y: calculated_angles["trunk_flexion_extension"] = trunk_angle_f_e
                else: calculated_angles["trunk_flexion_extension"] = -trunk_angle_f_e

        # 脚 (膝の角度)
        knee_angle = None
        if hip and knee and ankle:
            knee_angle = calculate_angle_3points_2d(knee, hip, ankle)
            calculated_angles["knee_angle"] = knee_angle
        
        # 上腕: (肩-肘)ベクトルと体幹線(肩-腰)の角度、または垂直線との角度
        upper_arm_angle_f_e = None
        if shoulder and elbow and mid_hip: # 体幹線(shoulder-mid_hip)を基準
            # 体幹ベクトル
            trunk_vector = calculate_vector_2d(mid_hip, shoulder)
            upper_arm_vector = calculate_vector_2d(shoulder, elbow)
            if trunk_vector and upper_arm_vector:
                # angle_between_vectors_2d は2つの独立したベクトル間の角度を計算
                # ここでは、(shoulder-mid_hip) と (shoulder-elbow) の間の角度を計算したい
                upper_arm_angle_f_e = calculate_angle_3points_2d(shoulder, mid_hip, elbow)
                # 0度が体側、前方が正、後方が負。 elbow.x と shoulder.x の関係で判断。
                # 撮影方向によるので、Y座標で判断する方が安定。
                # elbow.y < shoulder.y なら前方挙上。
                if upper_arm_angle_f_e is not None and elbow and shoulder:
                    if elbow.y < shoulder.y: calculated_angles["upper_arm_flexion_extension"] = upper_arm_angle_f_e
                    else: calculated_angles["upper_arm_flexion_extension"] = -upper_arm_angle_f_e # 体側または後方
        elif shoulder and elbow: # 垂直線基準の場合
            upper_arm_angle_f_e = calculate_angle_with_vertical(shoulder, elbow)
            if upper_arm_angle_f_e is not None and elbow and shoulder:
                if elbow.y < shoulder.y: calculated_angles["upper_arm_flexion_extension"] = upper_arm_angle_f_e
                else: calculated_angles["upper_arm_flexion_extension"] = -upper_arm_angle_f_e

        # 前腕 (肘の角度)
        lower_arm_angle_flex = None
        if shoulder and elbow and wrist:
            lower_arm_angle_flex = calculate_angle_3points_2d(elbow, shoulder, wrist)
            calculated_angles["lower_arm_flexion"] = lower_arm_angle_flex

        # 手首: (肘-手首)ベクトルと(手首-指先)ベクトルの角度 (指先がないので、手首の屈曲/伸展は難しい)
        # 手首の角度は、前腕のラインと手のひらのラインの角度。
        # (Elbow-Wrist) と (Wrist-Hand_Landmark) が必要。Hand Landmarkがない場合は推定困難。
        # ここでは仮に (Elbow-Wrist) と水平線の角度などで代用するか、手入力に頼る。
        # 既存コードでは wrist_angle_f_e があったので、何らかの方法で計算していたはず。
        # ここではNoneとしておく。
        wrist_angle_f_e = None 
        calculated_angles["wrist_flexion_extension"] = wrist_angle_f_e

        # --- スコア計算 ---
        neck_s = get_neck_score(calculated_angles.get("neck_flexion_extension"), manual.neck_lateral_bending, manual.neck_rotation)
        trunk_s = get_trunk_score(calculated_angles.get("trunk_flexion_extension"), manual.trunk_lateral_bending, manual.trunk_rotation)
        legs_s = get_legs_score(calculated_angles.get("knee_angle")) # 他のパラメータも考慮する必要あり
        
        table_a_score = get_table_a_score(trunk_s, neck_s, legs_s)
        load_score = manual.load_force # 0, 1, or 2
        if manual.shock_force: load_score +=1 # 衝撃力で+1
        final_table_a_score = table_a_score + load_score

        upper_arm_s = get_upper_arm_score(calculated_angles.get("upper_arm_flexion_extension"), manual.shoulder_raised, manual.upper_arm_abduction)
        lower_arm_s = get_lower_arm_score(calculated_angles.get("lower_arm_flexion"), manual.forearm_rotation)
        wrist_s = get_wrist_score(calculated_angles.get("wrist_flexion_extension"), manual.wrist_ulnar_radial_deviation, False) # 手首のねじりはPoseからは困難

        table_b_score = get_table_b_score(upper_arm_s, lower_arm_s, wrist_s)
        coupling_s = manual.coupling_score
        final_table_b_score = table_b_score + coupling_s

        table_c_s = get_table_c_score(final_table_a_score, final_table_b_score)
        activity_s = manual.activity_score # 0, 1, or 2 (REBAでは+1, +2, +3だが、入力は0,1,2)
        # REBAのActivity Scoreは1,2,3。入力が0,1,2なら+1する。
        # ただし、REBAシートでは「1つ以上の部位が静的(>1分保持)」「繰り返し(>4回/分)」「急速な大きな姿勢変化」で+1
        # ここではユーザー入力の activity_score をそのまま加算する（0, 1, or 2）
        # REBAの最終スコアは Table C Score + Activity Score (Activityが1点なら+1)
        # ユーザー入力が0,1,2で、それぞれ0点、1点、2点加算と解釈。
        final_reba_score = table_c_s + activity_s

        risk_level_str = "無視できるリスク"
        is_high_risk_flag = False
        if final_reba_score >= 11:
            risk_level_str = "超高リスク・即時改善"
            is_high_risk_flag = True
        elif final_reba_score >= 8:
            risk_level_str = "高リスク・要調査・改善"
            is_high_risk_flag = True
        elif final_reba_score >= 4:
            risk_level_str = "中リスク・要調査・変更の可能性"
        elif final_reba_score >= 2:
            risk_level_str = "低リスク・変更の必要性低い"

        return RebaScoreResponse(
            reba_score=float(final_reba_score),
            risk_level=risk_level_str,
            neck_score=neck_s,
            trunk_score=trunk_s,
            legs_score=legs_s,
            table_a_score=final_table_a_score,
            upper_arm_score=upper_arm_s,
            lower_arm_score=lower_arm_s,
            wrist_score=wrist_s,
            table_b_score=final_table_b_score,
            table_c_score=table_c_s,
            is_high_risk=is_high_risk_flag,
            angles=calculated_angles
        )

    except Exception as e:
        print(f"Error during REBA calculation: {e}")
        traceback.print_exc()
        return RebaScoreResponse(
            reba_score=None,
            risk_level="エラー",
            error_message=str(e),
            angles=calculated_angles
        )

# 開発用サーバー起動: uvicorn main:app --reload --port 8001
# (reba_app_v2/backend ディレクトリで実行)

