all: ff-convert.1 ff-similar.1 ff-train.1

clean:
	rm -f *.1

%1: %1.md
	pandoc -s -w man -o $@ $<
