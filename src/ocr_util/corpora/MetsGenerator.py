from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Final

from lxml import etree
from lxml.etree import XMLParser, _Element, _ElementTree

from ocr_util.corpora.common import MetsExtract, MetsGeneratorResource, MetsResource, GT_METS_FILEGROUP


class MetsGenerator:

    @staticmethod
    def __find_dmd_id(element: _Element) -> str:
        parent: _Element = element.getparent()
        type_: str = parent.get("TYPE")
        dmd_id: str = parent.get("DMDID")
        if dmd_id is not None and (type_ == 'monograph' or type_ == 'volume'):
            return dmd_id
        return MetsGenerator.__find_dmd_id(parent)

    __METS_FILE_NAME: Final[str] = "mets.xml"
    __INDENT: Final[int] = 4

    def __init__(self, out_dir: Path, generator_resources: list[MetsGeneratorResource], corpus_label: str = "Ground Truth Corpus") -> None:
        self.__out_dir: Final[Path] = out_dir
        self.__generator_resources: Final[list[MetsGeneratorResource]] = generator_resources
        # Template file is located in the same directory as this module
        template_path = Path(__file__).parent / 'template.corpus.xml'
        self.__mets_xml_document: Final[_ElementTree] = etree.parse(
            template_path,
            XMLParser(remove_blank_text=True)
        )
        self.__doc_root: Final[_Element] = self.__mets_xml_document.getroot()
        # Extract namespaces directly using xpath
        self.__nsmap: Final[dict[str, str]] = {
            ns[0]: ns[1] for ns in self.__doc_root.xpath('//namespace::*') if ns[0]
        }

        # GET SECTION ROOT NODES FROM DOC
        self.__physroot: Final[_Element] = self.__doc_root.find(
            f'.//mets:div[@ID="physroot"]',
            self.__nsmap
        )
        self.__logroot: Final[_Element] = self.__doc_root.find(
            f'.//mets:div[@ID="logroot"]',
            self.__nsmap
        )
        # Set the corpus label from parameter
        self.__logroot.set('LABEL', corpus_label)
        self.__struct_link_root: Final[_Element] = self.__doc_root.find(
            f'.//mets:structLink',
            self.__nsmap
        )
        self.__file_group_image: Final[_Element] = self.__doc_root.find(
            f'.//mets:fileGrp[@USE="OCR-D-IMG"]',
            self.__nsmap
        )
        self.__file_group_fulltext: Final[_Element] = self.__doc_root.find(
            f'.//mets:fileGrp[@USE="FULLTEXT"]',
            self.__nsmap
        )
        self.__file_group_fulltext.set('USE', GT_METS_FILEGROUP)

    def run(self) -> MetsResource:

        # EXTRACT DATA FROM ORIG METS AND INSERT
        total: int = len(self.__generator_resources)
        for i, generator_resource in enumerate(self.__generator_resources):
            print(f'Process {i + 1} of {total} mets files - {generator_resource.mets.identifier_urn}')
            extract: MetsExtract = self.__get_mets_data(
                index=i,
                page_urn=generator_resource.mets.identifier_urn,
                gt_file_path=generator_resource.gt.file_path,
                orig_mets_file_path=generator_resource.mets.file_path,
            )
            self.__physroot.append(extract.phys_div)
            self.__file_group_image.append(extract.file_image)
            self.__file_group_fulltext.append(extract.file_fulltext)
            self.__struct_link_root.append(extract.sm_link)
            self.__logroot.append(extract.log_div)
            if extract.dmd_sec is not None:
                idx: int = len(self.__doc_root.findall('.//mets:dmdSec', namespaces=self.__nsmap))
                self.__doc_root.insert(idx, extract.dmd_sec)

        phys_divs_with_order_attrib: list[_Element] = self.__physroot.findall(
            './/mets:div[@ORDER]',
            self.__nsmap
        )
        for i, elm in enumerate(phys_divs_with_order_attrib):
            elm.set('ORDER', str(i + 1))

        log_divs_with_order_attrib: list[_Element] = self.__logroot.findall(
            './/mets:div[@ORDER]',
            self.__nsmap
        )
        for i, elm in enumerate(log_divs_with_order_attrib):
            elm.set('ORDER', str(i + 1))

        # FORMAT
        etree.indent(self.__doc_root, space=(" " * MetsGenerator.__INDENT))

        # SAVE
        out_file: Path = self.__out_dir.joinpath(MetsGenerator.__METS_FILE_NAME)
        self.__mets_xml_document.write(
            out_file,
            xml_declaration=True,
            pretty_print=True,
            encoding=self.__mets_xml_document.docinfo.encoding
        )

        return MetsResource(
            identifier_urn="gt_2_mets",
            file_path=out_file
        )

    # #############################  SUB ##################################

    def __get_mets_data(
            self,
            index: int,
            page_urn: str,
            gt_file_path: Path,
            orig_mets_file_path: Path,
    ) -> MetsExtract:
        doc: _ElementTree = etree.parse(orig_mets_file_path, XMLParser(remove_blank_text=True))
        doc_root = doc.getroot()
        # Extract namespaces directly using xpath
        nsmap: dict[str, str] = {
            ns[0]: ns[1] for ns in doc_root.xpath('//namespace::*') if ns[0]
        }

        # PHYS
        phys_div: _Element = doc_root.find(f'.//mets:div[@CONTENTIDS="{page_urn}"]', nsmap)
        try:
            del phys_div.attrib['ORDERLABEL']
        except KeyError:
            pass

        # FILES
        file_pointers: list[_Element] = phys_div.findall('.//mets:fptr', namespaces=nsmap)
        # Remove elements from their parent
        for el in file_pointers:
            el.getparent().remove(el)
        files: list[_Element] = [
            doc_root.find(f'.//mets:file[@ID="{fptr.get("FILEID")}"]', nsmap)
            for fptr
            in file_pointers
        ]
        file_image: _Element = next(file for file in files if file.getparent().get('USE') == 'MAX')
        file_ptr_image: _Element = next(fptr for fptr in file_pointers if fptr.get('FILEID') == file_image.get('ID'))
        phys_div.append(file_ptr_image)
        file_fulltext_id: str = f'{GT_METS_FILEGROUP}-{(index + 1)}'
        file_ptr_fulltext: _Element = etree.Element(
            '{' + self.__nsmap['mets'] + '}fptr',
            attrib={'FILEID': file_fulltext_id}
        )
        phys_div.append(file_ptr_fulltext)
        file_fulltext: _Element = etree.Element(
            '{' + self.__nsmap['mets'] + '}file',
            attrib={
                'ID': file_fulltext_id,
                'MIMETYPE': 'application/vnd.prima.page+xml',
            }
        )
        file_fulltext.append(
            etree.Element(
                '{' + self.__nsmap['mets'] + '}FLocat',
                attrib={
                    "{" + self.__nsmap["xlink"] + "}href": str(gt_file_path.relative_to(self.__out_dir)),
                    'LOCTYPE': "OTHER",
                    'OTHERLOCTYPE': "FILE",
                }
            )
        )

        # LINK
        phys_id: str = phys_div.get('ID')
        sm_link: _Element = doc_root.find(f'.//mets:smLink[@xlink:to="{phys_id}"]', nsmap)

        # LOG
        log_id: str = sm_link.get("{" + self.__nsmap["xlink"] + "}from")
        log_div: _Element = doc_root.find(f'.//mets:div[@ID="{log_id}"]', nsmap)
        dmd_id: str = MetsGenerator.__find_dmd_id(log_div)
        log_div.set('DMDID', dmd_id)
        self.__file_group_fulltext.set('DMDID', dmd_id)
        try:
            del log_div.attrib['LABEL']
        except KeyError:
            pass
        # Remove all children elements
        for child in log_div.getchildren():
            child.getparent().remove(child)
        # log_div.append(file_ptr_image)

        dmd_sec: _Element | None = None

        mods_blocks: list[_Element] = doc_root.findall(f'.//mods:mods', nsmap)
        num_mods_blocks: int = len(mods_blocks)
        if num_mods_blocks > 1:
            raise Exception(
                f'METS file has more than one MODS block: found {num_mods_blocks} MODS blocks in {orig_mets_file_path}'
            )

        if self.__doc_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', namespaces=self.__nsmap) is None:
            orig_dmd_sec: _Element = doc_root.find(f'.//mets:dmdSec[@ID="{dmd_id}"]', nsmap)
            orig_mods_root: _Element = orig_dmd_sec.find(f'.//mods:mods', nsmap)

            dmd_sec = deepcopy(orig_dmd_sec)
            mods_root: _Element = dmd_sec.find(f'.//mods:mods', nsmap)
            # Remove all children elements
            for child in mods_root.getchildren():
                child.getparent().remove(child)

            mods_root.append(orig_mods_root.find(f'.//mods:titleInfo', nsmap))
            mods_root.extend(orig_mods_root.findall(f'.//mods:identifier', nsmap))
            mods_root.extend(orig_mods_root.findall(f'.//mods:language', nsmap))
            mods_root.extend(orig_mods_root.findall(f'.//mods:genre', nsmap))
            mods_root.append(orig_mods_root.find(f'.//mods:originInfo[@eventType="publication"]', nsmap))

        return MetsExtract(
            phys_div=phys_div,
            file_image=file_image,
            file_fulltext=file_fulltext,
            sm_link=sm_link,
            log_div=log_div,
            dmd_sec=dmd_sec
        )
