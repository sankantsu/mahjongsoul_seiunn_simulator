from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa

import streamlit as st


# ルール規定の定数にゃ
class MahjongConstants:
    NUM_SUUPAI = 9


class SeiunnConstants:
    BEAN_N_SELECT_PAI = 2
    BEAN_ENHANCE_RATIO = 1.5
    HIBIKI_ENHANCE_RATIO = 2.5
    WEIGHT_ENHANCE_RATIO = 1.8

    BEAN_POWER = 5
    WEIGHT_MAX_POWER = float(2**53 - 1)


# ソウズを 2 つ選ぶにゃ
def select_two() -> (int, int):
    i = random.randint(0, 8)
    j = i
    while j == i:
        j = random.randint(0, 8)
    return (i, j)


# シミュレーションのパラーメターにゃ
@dataclass
class SeiunnSimulationParams:
    n_haruna: int
    n_hibiki: int
    n_beans: int
    n_weight: int
    other_power: float = 1.0  # 花火等によるその他の倍率にゃ

    def bean_triggered_per_stage(self) -> int:
        n_enhance_per_bean = 1 + self.n_haruna
        return n_enhance_per_bean * self.n_beans

    def hibiki_enhance_ratio(self) -> float:
        return 1 + self.n_hibiki * (SeiunnConstants.HIBIKI_ENHANCE_RATIO - 1)

    def bean_enhance_ratio(self) -> float:
        return (
            1 + (SeiunnConstants.BEAN_ENHANCE_RATIO - 1) * self.hibiki_enhance_ratio()
        )

    def average_enhance_ratio_per_stage(self) -> float:
        n_avg_pai_enhance = (
            SeiunnConstants.BEAN_N_SELECT_PAI
            * self.bean_triggered_per_stage()
            / MahjongConstants.NUM_SUUPAI
        )
        r_beans = self.bean_enhance_ratio() ** n_avg_pai_enhance
        r_weight = SeiunnConstants.WEIGHT_ENHANCE_RATIO**self.n_weight
        return r_beans * r_weight

    # 1 ターンのスコアにかかる最大倍率にゃ
    def max_power(self) -> float:
        power_beans = SeiunnConstants.BEAN_POWER**self.n_beans
        power_weight = (
            SeiunnConstants.WEIGHT_MAX_POWER
        )  # とりあえずカンスト前提にするにゃ
        return power_beans * power_weight * self.other_power


# 星雲の志 1 回のシミュレーションにゃ
class SeiunnSimulator:
    # 初期化にゃ
    def __init__(self, params: SeiunnSimulationParams) -> None:
        self._params = params
        self._enhance_counts = [0] * MahjongConstants.NUM_SUUPAI

    @property
    def enhance_counts(self) -> list[int]:
        return self._enhance_counts

    # 1 ステージぶんの強化抽選を行うにゃ
    def simulate_stage(self) -> None:
        n_enhance_per_stage = self._params.bean_triggered_per_stage()
        for i in range(n_enhance_per_stage):
            i, j = select_two()
            self._enhance_counts[i] += 1
            self._enhance_counts[j] += 1

    # n_stage ステージ分のシミュレーションにゃ
    def run(self, n_stage: int) -> None:
        for i in range(n_stage):
            self.simulate_stage()


# 偏りで最もスコアが高くなった牌のスコアを集計するにゃ
class MaxScoreSimulator:
    def __init__(
        self,
        initial_scores: list[float],
        n_stage: int,
        seiunn_params: SeiunnSimulationParams,
        n_simulation: int = 10000,
    ) -> None:
        self._initial_scores = initial_scores.copy()
        self._n_stage = n_stage
        self._n_simulation = n_simulation
        self._seiunn_params = seiunn_params

    def run(self) -> list[float]:
        results = []

        progress_bar = st.progress(0, text="Running simulation...")
        for i in range(self._n_simulation):
            if i % 1000 == 0:
                progress_bar.progress(i / self._n_simulation, "Running simulation...")

            # 星雲 1 回の実行にゃ
            seiunn_sim = SeiunnSimulator(self._seiunn_params)
            seiunn_sim.run(self._n_stage)

            # 牌スコア計算にゃ
            enhance_counts = seiunn_sim.enhance_counts
            scores = self._initial_scores.copy()
            assert len(self._initial_scores) == len(enhance_counts)

            for i in range(len(self._initial_scores)):
                ratio = self._seiunn_params.bean_enhance_ratio()
                scores[i] *= ratio ** enhance_counts[i]

            # 最大スコアだけ使うにゃ
            max_score = max(scores)
            results.append(max_score)

        progress_bar.empty()

        return results


# ヒストグラムで分布を書くにゃ
# Best/Worst, Top 10%, 20%, ... のスコアも表示するにゃ
def visualize(
    max_scores: list[float], xlabel: str | None = None, ylabel: str | None = None
) -> None:
    fig, ax = plt.subplots()

    max_scores = np.sort(max_scores)
    log_max_scores = np.log10(max_scores)

    nbins = 20  # グラフの長方形の数にゃ
    ax.hist(log_max_scores, bins=nbins)

    n_simulation = len(max_scores)
    st.write(f"Best: {max_scores[-1]:.3e}")
    st.write(f"Worst: {max_scores[0]:.3e}")

    labels = []
    values = []
    for i in range(1, 10):
        idx = int(n_simulation * (1 - 0.1 * i))
        labels.append(f"Top {i*10} %")
        values.append(f"{max_scores[idx]:.3e}")
    df = pd.DataFrame({"割合": labels, "スコア": values})
    st.dataframe(
        df,
        hide_index=True,
    )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    st.pyplot(fig)


def main():
    st.title("青雲の志 簡易スコアシミュレータ")

    st.markdown("""
    ## 概要

    青雲の志 (2025年1月版) のデッキ構成に対して、スコアの伸びや最終スコアをシミュレーションするにゃ

    ## 使い方

    初期スコアとデッキ構成を入力すると、最大牌スコアや最終的な火力をシミュレーションできるにゃ
    偏りがどれぐらいあるのかの確認にも役立つはずにゃ
    """)

    st.markdown("""
    ### 初期スコア

    デッキ完成時点など、初期値となる時点での牌のスコアを入力するにゃ
    """)

    # 牌スコアの初期値にゃ
    # 初期値の偏りはこのへんで調整するにゃ
    initial_score = 10**6
    pai_list = [f"{i}s" for i in range(1, MahjongConstants.NUM_SUUPAI + 1)]
    df_initial_scores = pd.DataFrame(
        {
            "牌": pai_list,
            "初期スコア": [initial_score] * MahjongConstants.NUM_SUUPAI,
        }
    )

    if "df_initial_scores" not in st.session_state:
        st.session_state["df_initial_scores"] = df_initial_scores

    with st.form("set_all_initial_scores"):
        new_initial_score = st.number_input("牌スコアの初期値", value=10**6)
        set_all = st.form_submit_button("まとめて設定する")

    if set_all:
        st.session_state["df_initial_scores"]["初期スコア"] = [new_initial_score] * MahjongConstants.NUM_SUUPAI

    st.data_editor(st.session_state["df_initial_scores"], hide_index=True)

    # Ex 道中で使うデッキ構成にゃ
    n_haruna = st.number_input(
        "陽菜+ の数", value=1, min_value=0, max_value=8, key="haruna"
    )
    n_hibiki = st.number_input(
        "響+ の数", value=1, min_value=0, max_value=8, key="hibiki"
    )
    n_beans = st.number_input(
        "豆(木)+ の数", value=5, min_value=0, max_value=8, key="beans"
    )
    n_weight = st.number_input(
        "ウェイト+ の数", value=1, min_value=0, max_value=8, key="weight"
    )

    # シミュレーションのステージ数や試行回数にゃ
    n_stage = st.number_input(
        "ステージ数", value=55, min_value=1, max_value=300, key="stage"
    )
    n_simulation = st.number_input(
        "シミュレーション回数",
        value=10000,
        min_value=1,
        max_value=1000000,
        key="simulation",
    )

    seiunn_params = SeiunnSimulationParams(
        # デッキ構成によってここの数字を変えるにゃ
        n_haruna=n_haruna,
        n_hibiki=n_hibiki,
        n_beans=n_beans,
        n_weight=n_weight,
    )

    # 1 ターンあたりの平均的な成長率目安にゃ
    avg_enhance_ratio = seiunn_params.average_enhance_ratio_per_stage()
    st.write(f"1 ターンあたりの平均成長率: {avg_enhance_ratio}")

    # シミュレーション実行にゃ
    initial_scores = list(st.session_state["df_initial_scores"]["初期スコア"])
    simulator = MaxScoreSimulator(
        initial_scores=initial_scores,
        n_stage=n_stage,
        seiunn_params=seiunn_params,
        n_simulation=n_simulation,
    )
    max_scores = np.array(simulator.run())

    # 牌スコア表示にゃ
    st.markdown("""
    ### 最も育った牌のスコア分布

    9 牌の中でも最も高スコアに成長した牌のスコアの分布にゃ
    """)

    xlabel = "一番育った牌のスコア(log10)"
    ylabel = f"出現数 / {n_simulation}"
    visualize(max_scores, xlabel=xlabel, ylabel=ylabel)

    # 最終火力計算にゃ (デッキ交換した後のパラメータをつくるにゃ)
    st.markdown("""
    ### 最終デッキ構成 (カード入れかえ後)

    Ex 終盤でカード入れかえを行なった後のデッキを入力するにゃ
    嵐星入れるときは効果複製対象のカードが 1 枚増えたと思って入力するにゃ

    花火等の他の倍率がさらにある場合は「他の倍率」に入れるにゃ
    """)
    n_haruna_final = st.number_input(
        "陽菜+ の数", value=0, min_value=0, max_value=8, key="haruna_final"
    )
    n_hibiki_final = st.number_input(
        "響+ の数", value=0, min_value=0, max_value=8, key="hibiki_final"
    )
    n_beans_final = st.number_input(
        "豆(木)+ の数", value=4, min_value=0, max_value=8, key="beans_final"
    )
    n_weight_final = st.number_input(
        "ウェイト+ の数", value=2, min_value=0, max_value=8, key="weight_final"
    )
    other_power = st.number_input(
        "他の倍率", value=45, min_value=1, max_value=10**13, key="other_power"
    )

    st.markdown("""
    ### 翻数

    最終和了で想定する翻数にゃ
    """)
    n_fan = st.number_input("翻数", value=25, min_value=1, max_value=10**13, key="fan")

    final_seiunn_params = SeiunnSimulationParams(
        # 例: 陽菜 -> 嵐星交換で火力上昇させたケースを想定するにゃ
        # この場合嵐星はウェイト換算にゃ
        n_haruna=n_haruna_final,
        n_hibiki=n_hibiki_final,
        n_beans=n_beans_final,
        n_weight=n_weight_final,
        other_power=other_power,  # 花火とか、豆ウェイト以外の火力にゃ
    )
    # 1 番高い牌を 1 枚つかった和了のスコアにゃ
    powers = final_seiunn_params.max_power() * n_fan * max_scores

    st.markdown("""
    ### 最終火力の分布

    最も高い牌を 1 枚使って和了した際のスコア分布の目安にゃ (他の牌のスコアは考慮しないにゃ)
    """)

    xlabel = "最終火力 (log10)"
    ylabel = f"出現数 / {n_simulation}"
    visualize(powers, xlabel=xlabel, ylabel=ylabel)


if __name__ == "__main__":
    main()
