import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, t

def main():
    param_nums = [991995, 1156636, 1314120, 1481652, 1653866, 1817930, 1964108, 2122007, 2279711, 2443067, 2612075,
                  2779778, 2928578, 3083068, 3247185, 3409755, 3574350, 3736748, 3908963, 4058757, 4231690, 4387487,
                  4537211, 4689447, 4872926, 5030648, 5190882, 5353628, 5518886, 5686656, 5825435, 5997767, 6162250,
                  6304105, 6472707, 6620711, 6793468, 6942372, 7119248, 7274427, 7455458, 7611411, 7760207, 7947148,
                  8108136, 8261687, 8413707, 8570111, 8724930, 8884187, 9041805, 9203915, 9364332, 9529295, 9692511,
                  9860327, 10026342, 10183690, 10352393, 10480242, 10654715, 10813533, 10990748, 11122467, 11298745,
                  11465740, 11644706, 11780277, 11950780, 12133478, 12271857, 12445868, 12586011, 12773485, 12951004,
                  13093955, 13269953, 13399368, 13544767, 13727548, 13874711, 14055863, 14204772, 14391938, 14526700,
                  14678075, 14864381, 15017502, 15155155, 15348460, 15504047, 15643907, 15836226, 15977573, 16119548,
                  16278987, 16422292, 16623490, 16768300, 16930907, 17077047, 17277956, 17425583, 17591340, 17740297,
                  17889882, 18099850, 18250940, 18402658, 18572987, 18726035, 18879711, 19034015, 19246091, 19344507,
                  19500695, 19657511, 19814955, 19973027, 20131727, 20291055, 20451011, 20670818, 20832261, 20994332,
                  21157031, 21320358, 21484313, 21648896, 21814107, 21918875, 22085111, 22251975, 22419467, 22567754,
                  22736428, 22905730, 23008186, 23178495, 23349432, 23520997, 23693190, 23866011, 24018998, 24193001,
                  24367632, 24542891, 24654011, 24830295, 25007207]
    accuracy_difs = [0.7642, 0.7502, 0.7201, 0.7358, 0.7463, 0.7685, 0.7324999999999999, 0.7377, 0.7495, 0.7769,
                     0.7424999999999999, 0.7594, 0.7481, 0.7738, 0.7501, 0.7267, 0.7404999999999999, 0.7334, 0.7632,
                     0.7677, 0.7548, 0.7339, 0.7427, 0.7547, 0.7316, 0.7497, 0.7655000000000001, 0.7383, 0.7448, 0.7583,
                     0.7481, 0.7711, 0.716, 0.7488, 0.7484, 0.7464999999999999, 0.7498, 0.7816, 0.765, 0.7643, 0.7449,
                     0.7603, 0.7178, 0.7343, 0.7494000000000001, 0.749, 0.776, 0.7762, 0.7676000000000001, 0.7677,
                     0.7389, 0.77, 0.7457, 0.7531, 0.765, 0.7574, 0.7423, 0.7185, 0.7567, 0.7178, 0.7686,
                     0.7413000000000001, 0.7619, 0.7582, 0.7555000000000001, 0.7503, 0.7431, 0.7401, 0.7537, 0.7462,
                     0.7536, 0.7624, 0.7623, 0.7605999999999999, 0.7709, 0.7384999999999999, 0.7222999999999999, 0.7681,
                     0.7861, 0.7464, 0.7629, 0.7519, 0.7547, 0.7496, 0.744, 0.744, 0.7778, 0.7522, 0.8039000000000001,
                     0.752, 0.7453000000000001, 0.7702, 0.7646999999999999, 0.783, 0.7416, 0.7585, 0.762, 0.7814,
                     0.7512, 0.7684, 0.748, 0.7671, 0.7585999999999999, 0.754, 0.7658, 0.7834, 0.7866, 0.7443,
                     0.7525999999999999, 0.777, 0.7677, 0.7298, 0.7322, 0.7612, 0.7307, 0.7613, 0.7302, 0.7481, 0.7509,
                     0.7725, 0.7952, 0.7881, 0.7372000000000001, 0.7392000000000001, 0.7605, 0.7665, 0.7831,
                     0.7655000000000001, 0.7812, 0.7388, 0.7749, 0.7238, 0.738, 0.7686, 0.8055, 0.7818, 0.7743,
                     0.7284999999999999, 0.7603, 0.7842, 0.7418, 0.7992, 0.7622, 0.7663, 0.7622, 0.7711, 0.7671,
                     0.7605999999999999, 0.7835, 0.782]

    print(param_nums)
    print(accuracy_difs)

    scale_factor = 10_000_000

    # Create scaled DataFrame
    df = pd.DataFrame({
        'x': np.array(param_nums) / scale_factor,
        'y': accuracy_difs
    }, columns=['x', 'y'])

    # Perform linear regression
    result = linregress(df['x'], df['y'], alternative='greater')

    # Unpack regression results
    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue
    r_squared = r_value ** 2
    stderr = result.stderr
    t_stat = slope / stderr
    p_value = result.pvalue
    n = len(df)
    df_error = n - 2

    # 95% CI for slope
    alpha = 0.05
    t_crit = t.ppf(1 - alpha / 2, df_error)
    ci_low = slope - t_crit * stderr
    ci_high = slope + t_crit * stderr

    # Print Minitab-style regression output
    print("Regression Analysis (x scaled by 10 million parameters)")
    print(f"{'Predictor':<12}{'Coef':>10}{'SE Coef':>12}{'T':>10}{'P':>10}")
    print(f"{'Constant':<12}{intercept:10.4f}{'':>12}")
    print(f"{'x':<12}{slope:10.4f}{stderr:12.4f}{t_stat:10.4f}{p_value:10.4f}")
    print()
    print(f"S = {np.sqrt(result.stderr ** 2 * df_error / (n - 1)):.4f}   R-sq = {r_squared * 100:.2f}%")
    print(f"95% CI for slope: ({ci_low:.4f}, {ci_high:.4f})")

    # Plot
    plt.figure()
    plt.grid(True, zorder=1)
    plt.scatter(x=df['x'], y=df['y'], label='Data Points', color='green', s=20, zorder=2)
    x_vals = np.linspace(df['x'].min(), df['x'].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', label=fr'LSRL: $\hat y = {slope:.4f}x + {intercept:.4f}$', zorder=3)

    plt.xlabel('Parameter Count (x10 million)')
    plt.ylabel('Train Accuracy — Test Accuracy')
    plt.title('Layer Size vs Difference in Accuracy')
    plt.legend()
    plt.savefig("Scatter+LSRL.png", dpi=300)
    plt.show()

    y_pred = intercept + slope * df['x']
    residuals = df['y'] - y_pred
    print(*(str(residuals.tolist()[i]) + "\n" for i in range(len(y_pred))))
    plt.figure()
    plt.grid(True, zorder=0)
    plt.axhline(0, color='black', zorder=1)
    plt.scatter(df['x'], residuals, color='green', s=20, zorder=2)
    plt.xlabel('Parameter Count (x10 million)')
    plt.ylabel('Residual')
    plt.title('Residual plot')
    plt.savefig("Resid.png", dpi=300)
    plt.show()

    plt.figure()
    plt.hist(residuals, bins=20, edgecolor='black')
    xticks = np.round(np.linspace(min(residuals), max(residuals), num=15), 2)
    plt.xticks(xticks)
    plt.xlabel('Residual')
    plt.title('Histogram of Residuals')
    plt.savefig("Resid Hist.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()