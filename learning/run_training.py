from IQN.utilities.config_manager import CFG
import subprocess


def main():
    W = CFG.WORLDS
    P = CFG.policy_type
    algorithm_name = CFG.algorithm_name
    is_continued = CFG.is_continued

    # subprocess.call(['useradd', '-m', '-g', _primarygroup, '-G', _secondarygroup, '-u', _userid, _username])

    # p1 =  subprocess.call(["python", "C:\scripts\other.py"])
    # p2 = subprocess.call(["python", "C:\scripts\other.py"])
    # subprocess.run(["python", "C:\scripts\other.py"])


    for iworld in (W if isinstance(W,list) else [W]):
        for policy in (P if isinstance(P,list) else [P]):
            # is_loaded = True if policy in ['Averse', 'Seeking'] else False
            is_loaded = False
            run_IDQN(iworld, algorithm_name=algorithm_name, policy_type=policy, is_loaded=is_loaded, continued=is_continued)



