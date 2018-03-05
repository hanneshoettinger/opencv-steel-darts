from Calibration import calibrate


def kickoff():
    global cal_data_l
    global cal_data_r

    cal_data_l, cal_data_r = calibrate()


if __name__ == '__main__':
    kickoff()
