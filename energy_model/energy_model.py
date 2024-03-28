class EnergyReport:
    def __init__(self):
        self.systolic_array_mac_count=0
        self.vector_add_count=0
        self.vector_mul_count=0
        self.vector_div_count=0
        self.l1_count=0
        self.l2_count=0
        self.mem_count=0
    
    def reset(self):
        self.systolic_array_mac_count=0
        self.vector_add_count=0
        self.vector_mul_count=0
        self.vector_div_count=0
        self.l1_count=0
        self.l2_count=0
        self.mem_count=0