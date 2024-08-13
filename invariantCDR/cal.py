def calculate_average_(numbers):
    if not numbers:
        return 0  # 如果列表为空，返回0
    return sum(numbers) / len(numbers)

def calculate_average(name, hit, ndcg):
    avg_hit = calculate_average_(hit)
    avg_ndcg = calculate_average_(ndcg)
    print("{}\thit@10:{:.2f}, ndcg@10:{:.2f}".format(name, avg_hit, avg_ndcg))
    
cdrib_cloth_hit = [13.24, 13.15, 12.86, 12.80, 13.15]
cdrib_cloth_ndcg = [7.17, 7.22, 6.91, 6.83, 7.26]
cdrib_sport_hit = [10.67, 10.74, 11.17, 11.20, 11.73]
cdrib_sport_ndcg = [5.36, 5.54, 5.83, 5.98, 6.19]
print("----------------------CDRIB cloth_sport-------------------------")
average = calculate_average("CDRIB_cloth", cdrib_cloth_hit, cdrib_cloth_ndcg)
average = calculate_average("CDRIB_sport", cdrib_sport_hit, cdrib_sport_ndcg)

DisCo_cloth_hit = [12.55, 12.86, 11.15, 12.39, 13.97]
DisCo_cloth_ndcg = [6.42, 7.08, 6.23, 6.88, 7.42]
DisCo_sport_hit = [10.67, 10.76, 10.20, 11.20, 10.59]
DisCo_sport_ndcg = [5.50, 5.35, 5.21, 5.98, 5.40]
print("----------------------DisCo cloth_sport-------------------------")
average = calculate_average("DisCo_cloth", DisCo_cloth_hit, DisCo_cloth_ndcg)
average = calculate_average("DisCo_sport", DisCo_sport_hit, DisCo_sport_ndcg)




cdrib_game_hit = [9.05, 10.07, 9.13, 10.28, 9.92]
cdrib_game_ndcg = [4.62, 5.36, 5.34, 5.12, 5.08]
cdrib_video_hit = [12.20, 12.89, 12.61, 13.45, 12.14]
cdrib_video_ndcg = [6.19, 6.66, 6.83, 6.95, 6.66]
print("----------------------CDRIB game_video-------------------------")
average = calculate_average("CDRIB_game", cdrib_game_hit, cdrib_game_ndcg)
average = calculate_average("CDRIB_video", cdrib_video_hit, cdrib_video_ndcg)

DisCo_game_hit = [9.85, 8.98, 9.56, 10.12, 9.20]
DisCo_game_ndcg = [4.71, 4.59, 4.44, 4.99, 4.75]
DisCo_video_hit = [13.31, 13.52, 13.31, 13.45, 13.31]
DisCo_video_ndcg = [6.99, 6.66, 7.09, 6.55, 6.99]
print("----------------------DisCo game_video-------------------------")
average = calculate_average("DisCo_game", DisCo_game_hit, DisCo_game_ndcg)
average = calculate_average("DisCo_video", DisCo_video_hit, DisCo_video_ndcg)





UniCDR_cell_hit = [14.85, 14.42, 14.48, 14.04, 13.55]
UniCDR_cell_ndcg = [9.53, 8.73, 8.98, 8.59, 8.53]
UniCDR_electronic_hit = [15.16, 15.50, 16.20, 16.33, 15.17]
UniCDR_electronic_ndcg = [9.54, 8.96, 9.49, 9.49, 8.76]
print("----------------------UniCDR cell_electronic-------------------------")
average = calculate_average("UniCDR_cell", UniCDR_cell_hit, UniCDR_cell_ndcg)
average = calculate_average("UniCDR_eletronic", UniCDR_electronic_hit, UniCDR_electronic_ndcg)