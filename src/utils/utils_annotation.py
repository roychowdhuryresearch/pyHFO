def calcuate_boundary(start:int, end:int, length:int, win_len=2000)->tuple:
            """_summary_
            Args:
            start (int): the start of the window
            end (int): the end of the window
            length (int): the length of the signal
            win_len (int, optional): the length of the window we want to extend the signal to . Defaults to 2000.

            Returns:
            tuple: the start and end of the extended window, and the new shifted start and end of the window
            """
            if start < win_len: 
                  return 0, int(win_len*2),int(start), int(end)
            if end > length - win_len:
                  #in this case int(length-win_len*2) is the new start, so minus the start and end by that
                  return int(length - win_len*2), length, int(start)-int(length-win_len*2), int(end)-int(length-win_len*2)
            window_start = int(0.5*(start + end - win_len))
            window_end = int(0.5*(start + end + win_len))
            relative_start = int(start - 0.5*(start + end - win_len))
            relative_end = int(end - 0.5*(start + end - win_len))
            return window_start, window_end, relative_start, relative_end