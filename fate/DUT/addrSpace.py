###############################################################
###       DEFINE THE ADDRESS SPACE     ###
###############################################################


class AddrSpace:

    def __init__(self, parameters):
        self.physical_mem_size = parameters.PHYSICAL_MEM_SIZE
        self.pe_space = parameters.PE_SPACE
        self.pe_program_space = parameters.PE_PROGRAM_SPACE
        self.pe_spill_space = parameters.PE_SPILL_SPACE
        self.system_space = parameters.SYSTEM_SPACE
        self.overall_pe_space = parameters.OVERALL_PE_SPACE

    def get_pe_base(self, pe):
        """
            Base address for the PE space.

            input: PE number
        """
        return self.pe_space * pe

    def get_pe_program_base(self,pe):
        """
            Return the program base address for a PE.

            input: PE number
        """
        return self.get_pe_base(pe) + 0

    def get_pe_spill_base(self,pe):
        """
            Return the register spill base address for a PE.

            input: PE number
        """
        return self.get_pe_base(pe) + self.pe_program_space

    def get_pe_reserved_base(self,pe):
        """
            Return the program base address for a PE.

            input: PE number
        """
        return self.get_pe_base(pe) + self.pe_spill_space

    def get_shared_base(self):
        """
            Return the base address which is open to everyone
            to read and write.
        """
        return self.overall_pe_space + self.system_space
        