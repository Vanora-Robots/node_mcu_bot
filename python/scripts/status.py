


class Status:
    def showStats(self, centers):

        if self.std == None:
            return

        self.std.clear()
        self.std.addstr(0, 0, "Press Esc to Quit...")
        self.std.addstr(1, 0, "Total number of Leds found :{} \t ".format((len(centers))))
        i = 2
        for c in centers:
            i = i + 1
            self.std.addstr(i, 0, " X : {} \t {} \t".format(c[0], c[1]))

        self.std.refresh()