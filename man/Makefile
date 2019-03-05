all: ff-train.1

clean:
	rm -f *.1 *.5

%: %.md
	pandoc -s -w man -o $@ $<
