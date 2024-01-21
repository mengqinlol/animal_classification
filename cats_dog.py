import train_and_test as tt

if __name__ == "__main__":
    tt.train_dog(20)
    tt.train_cat(20)
    tt.train_bird(40)
    tt.train_first(5)
    tt.test_all_init()
    tt.train_all(20)
    print(tt.ans)