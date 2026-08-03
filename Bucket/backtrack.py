def choose_alternative(candidates, failed_index):
    for i in range(failed_index+1,len(candidates)):
        return candidates[i]
    return None
